import ctypes
import random
import time
from datetime import datetime

import numpy as np
from gym import spaces
from pyrep import PyRep
from pyrep.objects import VisionSensor, ProximitySensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from cffi import FFI

from DDPG.my_proximity_sensor import MyProximitySensor
from DDPG.running_min_max_normalizer import RunningMinMaxNormalizer
from DDPG.running_normalizer import RunningNormalizer


""" Class responsible for defining environment rules and simulation logic """
class HexapodEnv:
    def __init__(self, results_queue, disable_rendering=False, rewards=1):
        """ Initialize PyRep connection """
        self.pr = PyRep()
        self.pr.launch('red_box_scene_v9_flat.ttt', headless=disable_rendering)
        self.pr.start()

        """ Load all joints of hexapod """
        self.joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
        self.joints = [Joint(name) for name in self.joint_names]

        """ Define action and observation space """
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(53,), dtype=np.float32)

        self.vision_sensor = VisionSensor('Photo_sensor')

        self.distance_sensor = MyProximitySensor('Proximity_sensor')

        self.previous_distance = None

        self.initial_distance = 0

        self.previous_joint_positions = None

        self.initial_observation = None

        self.terminate = False

        self.velocity_towards_goal = None

        self.rewards = rewards

        self.robot = Object.get_object('hexapod')

        self.robot_body = Object.get_object('hexa_body')
        self.goal = Object.get_object('Target')
        self.terrain = Object.get_object("Terrain_shape")
        self.base_dimensions = Object.get_object("Plane").get_bounding_box()
        self.wall = Object.get_object("Walls")
        self.robot_feet = []
        self.robot_legs = []
        for i in range(6):
            self.robot_feet.append(Object.get_object(f"hexa_footTip{i}"))
            self.robot_legs.append(Object.get_object(f"hexa_link3_{i}"))

        self.max_tilt_penalization = np.deg2rad(75)
        self.max_tilt_termination = np.deg2rad(90)

        #self.reward_normalizer = RunningNormalizer(shape=(1,))
        self.target_range = 2000.0 / 300.0

        self.distance_normalizer = RunningNormalizer(shape=(1,))
        #self.velocity_normalizer = RunningNormalizer(shape=(1,))
        self.velocity_normalizer = RunningMinMaxNormalizer()
        #self.angular_velocity_normalizer = RunningMinMaxNormalizer()

        self.reward_min = float('inf')
        self.reward_max = float('-inf')

        self.current_distances = []
        self.current_velocities = []
        self.current_rewards = []

        # TODO 10 ?
        self.max_simulation_count = 10
        self.max_simulation_count_reset = 100

        self.T1 = 25_000  # stabilita → nohy
        self.T2 = 50_000  # nohy → pohyb
        self.T3 = 100_000  # pohyb → cíl

        self.phase_limit = 20_000
        self.current_phase = 3
        self.max_phase_num = 3

        self.w_stability = 1.0

        self.w_legs = 0

        self.w_motion = 0

        self.w_goal = 0

        self.current_global_step_count = 0

        self.results_queue = results_queue

        self.current_step = 0
        self.gait_period = 20

        self.fixed_robot_pos = None
        self.fixed_goal_pos = None


    """ Method responsible for setting new joint positions, advancing to the next state and returning relevant information about this action"""
    def set_joint_positions(self, action, episode):
        for joint, value in zip(self.joints, action):
            joint.set_joint_target_position(value)

    def step_and_update(self):
        self.pr.step()
        #self.current_step += 1
        #self._update_state()

    def get_step_results(self, episode, warmup = False):
        obs = self._get_observation(self.goal.get_position())
        done, goal_reached = self._check_termination()
        reward, leg_reward = self._calculate_reward(goal_reached, done, warmup)

        return obs, reward, done, goal_reached


    def all_joints_reached_targets(self, position_tolerance=0.05, velocity_tolerance=0.20):
        for index, j in enumerate(self.joints):
            pos = j.get_joint_position()
            target = j.get_joint_target_position()

            pos_not_synced = abs(pos - target) > position_tolerance

            joint_velocity = abs(j.get_joint_velocity())
            high_velocity = joint_velocity > velocity_tolerance

            if pos_not_synced and high_velocity:
                return False

        return True

    def _generate_new_position(self, min_x, max_x, min_y, max_y, offset):
        return random.uniform(min_x+offset, max_x-offset), random.uniform(min_y+offset, max_y-offset)

    def _randomly_place_robot_target(self, always_same_pos=False):
        #robot = Object.get_object("hexapod")
        robot_position = self.robot.get_position()
        target_position = self.goal.get_position()

        # base = Object.get_object("Terrain_shape")
        # base_dimensions = base.get_bounding_box()
        # wall = Object.get_object("Walls")

        stable_positions = True

        """ FOR STATIC"""
        #random.seed(1)

        def _move_object_to_new_position(affected_object, z_position, collision_object):
            while True:
                new_x, new_y = self._generate_new_position(self.base_dimensions[0], self.base_dimensions[1], self.base_dimensions[2], self.base_dimensions[3], 3.0)
                affected_object.set_position([new_x, new_y, z_position])
                self.pr.step()
                if affected_object.check_distance(collision_object) > 1.5:
                    break

        def _ensure_robot_goal_distance():
            while self.robot.check_distance(self.goal) <= 1.5:
                _move_object_to_new_position(self.robot, robot_position[2], self.goal)

            # detected_distance = self.distance_sensor.read()
            #
            # """ If the sensor measured something"""
            # if detected_distance != -1.0:
            #     return False
            # return True
            # #return not Object.get_object("hexapod").check_collision(Object.get_object("Terrain_shape"))

        """ ONLY FOR STATIC ROBOT - GOAL SPAWN LOCATION """
        if always_same_pos and self.fixed_robot_pos is not None and self.fixed_goal_pos is not None:
            self.robot.set_position(self.fixed_robot_pos)
            self.goal.set_position(self.fixed_goal_pos)
            self.pr.step()
            random.seed(None)
        else:
            """ Keep the original Z-coordinate, set new coordinates """

            _move_object_to_new_position(self.robot, robot_position[2], self.wall)
            _move_object_to_new_position(self.goal, robot_position[2], self.wall)

            """ Ensure that robot and goal are not position right next to each other"""
            _ensure_robot_goal_distance()

        if always_same_pos and self.fixed_robot_pos is None:
            self.fixed_robot_pos = self.robot.get_position()
            self.fixed_goal_pos = self.goal.get_position()
            self.results_queue.put((self.fixed_robot_pos, self.fixed_goal_pos))



        """ Wait until the robot and target reaches ground"""
        current_simulation_count = 0
        last_cube_pos, last_robot_pos = float("inf"), float("inf")
        current_cube_pos, current_robot_pos = self.goal.check_distance(self.terrain), self.robot.check_distance(self.terrain)
        while current_cube_pos < last_cube_pos or current_robot_pos < last_robot_pos:
            self.pr.step()
            current_simulation_count += 1
            #print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))

            if current_simulation_count >= self.max_simulation_count_reset:
                # print("Maximum simulation count reached")
                # print(self.robot.get_position())
                # print(self.distance_sensor.read())
                # print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))
                # print(self.goal.get_position())
                # print(robot.check_distance(Object.get_object("Terrain_shape")))

                #self.results_queue.put(f"Robot or Cube still not on the ground")
                stable_positions = False
                break

            last_cube_pos, last_robot_pos = current_cube_pos, current_robot_pos
            current_cube_pos, current_robot_pos = self.goal.check_distance(self.terrain), self.robot.check_distance(self.terrain)

        current_simulation_count = 0
        last_robot_pos = float("-inf")
        current_robot_pos = self.robot.check_distance(self.terrain)

        """ Wait for the robot to recover from the jump"""
        while current_robot_pos > last_robot_pos:
            self.pr.step()
            current_simulation_count += 1

            if current_simulation_count >= self.max_simulation_count_reset:
                # print("Maximum simulation count reached reverb")
                # print(self.robot.get_position())
                # print(self.distance_sensor.read())
                # print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))
                # print(self.goal.get_position())

                #self.results_queue.put(f"Robot still not recovered from the jump")
                stable_positions = False
                break

            last_robot_pos = current_robot_pos
            current_robot_pos = self.robot.check_distance(self.terrain)

        """ The target cube could be still in motion, need to ensure that its not moving when the simulation starts"""
        current_target_velocity, _ = self.goal.get_velocity()
        current_robot_body_velocity, _ = self.robot_body.get_velocity()

        absolute_target_velocity = [abs(vel) for vel in current_target_velocity]
        absolute_robot_body_velocity = [abs(vel) for vel in current_robot_body_velocity]

        current_simulation_count = 0
        while sum(absolute_target_velocity) > 0.001 or sum(absolute_robot_body_velocity) > 0.1:
            self.pr.step()

            current_simulation_count += 1
            if current_simulation_count >= self.max_simulation_count_reset:
                #self.results_queue.put(f"Cube or robot are still not stabilized")
                stable_positions = False
                break

            current_target_velocity, _ = self.goal.get_velocity()
            current_robot_body_velocity, _ = self.robot_body.get_velocity()

            absolute_target_velocity = [abs(vel) for vel in current_target_velocity]
            absolute_robot_body_velocity = [abs(vel) for vel in current_robot_body_velocity]




        return stable_positions



    """ Method responsible for resetting the simulation"""
    def reset(self):
        self.previous_distance = None
        self.previous_joint_positions = None
        self.current_step = 0

        for distance in self.current_distances:
            self.distance_normalizer.update(distance)

        for velocity in self.current_velocities:
            self.velocity_normalizer.update(velocity)

        # for reward in self.current_rewards:
        #     self.reward_normalizer.update(reward)

        #
        self.current_distances = []
        self.current_velocities = []
        # self.current_rewards = []

        self.pr.stop()

        self.pr.start()

        """ Randomly place robot and target into scene"""
        current_simulation_count = 0
        while True:
            have_correct_positions = self._randomly_place_robot_target(always_same_pos=False)
            current_simulation_count += 1
            if have_correct_positions:
                break

            self.pr.stop()

            self.pr.start()
            if current_simulation_count >= self.max_simulation_count:
                self.results_queue.put(f"ROBOT AND GOAL STILL problem")

        observation = self._get_observation(self.goal.get_position())

        # print(observation)
        # print("-------------")

        return observation

    def _update_state(self):
        self.robot = Object.get_object('hexa_base')
        self.robot_body = Object.get_object('hexa_body')
        self.goal = Object.get_object('Target')

        # print(self.robot.get_bounding_box())
        # print(self.robot_body.get_bounding_box())
        # print("--")

    """ Method responsible for handling observation space logic """
    def _get_observation(self, goal_pos):
        """ Get hexapod joint positions """
        num_joints = len(self.joints)
        # linear_velocities = np.zeros(num_joints*3)
        # angular_velocities = np.zeros(num_joints*3)
        joint_velocities = np.zeros(num_joints)
        positions = np.zeros(num_joints)

        """ Get both linear and angular joint velocities """
        for i, joint in enumerate(self.joints):
            # linear, angular = joint.get_velocity()
            #
            # linear_velocities[i * 3:(i + 1) * 3] = linear
            # angular_velocities[i * 3:(i + 1) * 3] = angular

            joint_vel = joint.get_joint_velocity()
            joint_velocities[i] = joint_vel

            positions[i] = joint.get_joint_position()


        """ Get relevant information about hexapod """
        robot_pos = self.robot_body.get_position()


        roll, pitch, yaw = self.robot_body.get_orientation()
        # --- Rotation: world -> body (yaw only) ---
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

        # --- Target direction (BODY FRAME) ---
        # target_vec_world = goal_pos - robot_pos
        # distance_to_target = np.linalg.norm(target_vec_world)
        # target_dir_world = target_vec_world / np.maximum(distance_to_target, 1e-8)
        # target_dir_body = R @ target_dir_world

        direction = goal_pos - robot_pos

        #direction[2] = 0.0
        # direction_norm = np.linalg.norm(direction)
        direction_to_goal_normalized = direction / np.maximum(np.linalg.norm(direction), 1e-8)

        #target_dir_velocity, _ = self._calculate_robot_velocity(self.goal.get_position(), self.robot_body.get_position())
        target_dir_velocity = self._calculate_robot_velocity_simple(self.goal.get_position(), self.robot_body.get_position())

        # --- Velocities (BODY FRAME) ---
        # linear_world, angular_world = self.robot_body.get_velocity()
        # linear_body = R @ linear_world
        # angular_body = R @ angular_world

        linear_world, angular_world = self.robot_body.get_velocity()

        # --- Orientation: REMOVE YAW ---
        hexa_orient = np.array([
            np.sin(roll), np.cos(roll),
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw)
        ])


        tilt_cos = self._get_robot_tilt()

        #linear, angular = self.robot.get_velocity()
        # target_direction = goal_pos - hexa_position
        # target_direction = target_direction / np.maximum(np.linalg.norm(target_direction), 1e-8)



        """ ROBOT LEGS"""
        # normal = self._get_normal_from_terrain()
        # foot_contacts = np.zeros(6)
        #
        # if normal is not None:
        #     for index, (leg, tip) in enumerate(zip(self.robot_legs, self.robot_feet)):
        #         current_leg_pos = leg.get_position()
        #         current_leg_tip_pos = tip.get_position()
        #         direction_of_the_leg = current_leg_pos - current_leg_tip_pos
        #         direction_of_the_leg /= np.maximum(np.linalg.norm(direction_of_the_leg), 1e-8)
        #         result = np.dot(direction_of_the_leg, normal)
        #         result = np.clip(result, 0, 1)
        #         foot_contacts[index] = result

        foot_contacts = np.zeros(6)

        for i in range(len(self.robot_feet)):
            d = self.robot_feet[i].check_distance(self.terrain)
            foot_contacts[i] = np.exp(-10 * d)


        body_height = self.robot_body.check_distance(self.terrain)

        phase = (self.current_step % self.gait_period) / self.gait_period
        phase_obs = np.array([
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase)
        ])

        #observation = np.concatenate((direction_to_goal_normalized, [target_dir_velocity], positions, joint_velocities, hexa_orient, linear_world, angular_world, foot_contacts, [body_height]))
        observation = np.concatenate((direction_to_goal_normalized, [target_dir_velocity], positions, joint_velocities,
                                      hexa_orient, linear_world, angular_world, [body_height]))

        return observation.astype(np.float32)

    """ Method responsible for calculating velocity of hexapod towards the goal"""
    def _calculate_robot_velocity_simple(self, goal_pos, robot_body_pos):
        direction = goal_pos - robot_body_pos

        #direction[2] = 0.0
        #direction_norm = np.linalg.norm(direction)
        direction_to_goal_normalized = direction / np.maximum(np.linalg.norm(direction), 1e-8)

        linear, _ = self.robot_body.get_velocity()

        #linear[2] = 0.0

        velocity_toward_goal = float(np.dot(linear, direction_to_goal_normalized))

        return velocity_toward_goal

    """ Method responsible for calculating velocity of hexapod towards the goal"""
    def _calculate_robot_velocity(self, goal_pos, robot_body_pos):
        roll, pitch, yaw = self.robot_body.get_orientation()
        # --- Rotation: world -> body (yaw only) ---
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

        # --- Target direction (BODY FRAME) ---
        target_vec_world = goal_pos - robot_body_pos
        distance_to_target = np.linalg.norm(target_vec_world)
        target_dir_world = target_vec_world / np.maximum(distance_to_target, 1e-8)
        target_dir_body = R @ target_dir_world

        # --- Velocities (BODY FRAME) ---
        linear_world, angular_world = self.robot_body.get_velocity()
        linear_body = R @ linear_world
        angular_body = R @ angular_world

        return float(np.dot(linear_body, target_dir_body)), np.linalg.norm(linear_body)
        # R_world_body = self.robot_body.get_matrix()[:3, :3]
        # R_body_world = R_world_body.T  # inverse
        #
        # # --- Goal direction in BODY frame ---
        # target_vec_world = goal_pos - robot_body_pos
        # target_vec_body = R_body_world @ target_vec_world
        #
        # # project to body horizontal plane (ignore body Z)
        # target_vec_body_xy = target_vec_body.copy()
        # target_vec_body_xy[2] = 0.0
        # target_norm = np.linalg.norm(target_vec_body_xy)
        # target_dir_body = target_vec_body_xy / np.maximum(target_norm, 1e-8)
        #
        # # --- Velocity in BODY frame ---
        # linear_world, _ = self.robot_body.get_velocity()
        # linear_body = R_body_world @ linear_world
        #
        # linear_body_xy = linear_body.copy()
        # linear_body_xy[2] = 0.0
        #
        # # --- Velocity toward goal ---
        # v_toward_goal = float(np.dot(linear_body_xy, target_dir_body))
        # speed_body = float(np.linalg.norm(linear_body_xy))
        #
        # return v_toward_goal, speed_body

    """ Scale reward to [-10; 10] range"""
    def _scale_reward(self, raw_value, min_value, max_value):
        new_min = -10
        new_max = 10
        new_interval = new_max - new_min
        return new_min + (raw_value - min_value) * (new_interval / (max_value - min_value))

    def _get_normal_from_terrain(self):
        _, _, _, normal = self.distance_sensor.read()

        if normal is not None:
            R_s = self.distance_sensor.get_matrix()[:3, :3]
            n = R_s @ normal

            # FIX: sensor Z is down → world Z is up
            n = -n

            n /= np.maximum(np.linalg.norm(n), 1e-8)
            return n

        return None

    def _get_robot_tilt(self):
        n = self._get_normal_from_terrain()
        if n is not None:
            R_r = self.robot_body.get_matrix()[:3, :3]
            z = R_r[:, 2]
            z /= np.maximum(np.linalg.norm(z), 1e-8)

            cos_theta = np.dot(z, n)
            return max(0, cos_theta)

        return 0

    def _smoothstep(self, x):
        return x * x * (-2 * x + 3)

    """ Method responsible for calculating reward based on the current state of hexapod """
    def _calculate_reward(self, goal_reached, done, warmup):
        # current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        # distance = 0
        #
        # if self.previous_distance is not None:
        #     distance = self.previous_distance - current_distance
        #
        # if distance != 0:
        #     self.current_distances.append(distance)
        #     #self.distance_normalizer.update(distance)
        # distance_reward = self.distance_normalizer.normalize(distance)[0]
        # distance_reward = np.clip(distance_reward, -3, 3)
        #
        #
        # self.previous_distance = current_distance

        #linear, _ = self.robot_body.get_velocity()
        #velocity, robot_speed = self._calculate_robot_velocity(self.goal.get_position(), self.robot_body.get_position())
        #velocity, _= self._calculate_robot_velocity(self.goal.get_position(), self.robot_body.get_position())
        velocity = self._calculate_robot_velocity_simple(self.goal.get_position(), self.robot_body.get_position())

        if warmup:
            self.velocity_normalizer.update(velocity)
        velocity_reward = self.velocity_normalizer.normalize(velocity)

        v_target = 0.3
        # velocity_reward = velocity_reward - v_target
        #
        # velocity_reward = 1.0 - (velocity_reward ** 2)

        if velocity_reward <= v_target:
            velocity_reward = velocity_reward / v_target
        else:
            velocity_reward = 1.0 - 5.0 * (velocity_reward - v_target)


        # """ Reward robot moving to goal"""
        # if robot_speed > 0.0:
        #     #correct_direction_reward = velocity / robot_speed
        #     correct_direction_reward = 1
        # else:
        #     correct_direction_reward = 0


        roll, pitch, _ = self.robot_body.get_orientation()
        tilt_reward, goal_reached_reward, fell_to_the_side_reward, body_height_reward = 0, 0, 0, 0
        alive_reward = 1

        fell_over = False

        leg_tip_reward = 0
        # for i in range(6):
        #     if self.robot_feet[i].check_distance(self.terrain) < 0.01:
        #         leg_tip_reward += 1/6

        scale = 10
        proximities = []
        leg_rewards = []
        for i in range(6):
            d = self.robot_feet[i].check_distance(self.terrain)

            """ We allow little measuring error"""
            if d < 0.05:
                d = 0
            proximity = np.exp(-scale * d)
            proximities.append(proximity)

            #leg_tip_reward += (1 / 6) * proximity - (1 / 6)

            d_max = 0.5  # distance where penalty saturates

            d = min(d, d_max)
            reward = -(1 / 6) * (d / d_max)

            leg_tip_reward += reward



        tilt = self._get_robot_tilt()
        """ To ensure -1; 1 range"""
        #tilt_reward = 2 * ((-1.0 + tilt) + 1) -1

        tilt_reward = tilt - 1.0

        """ ROBOT LEGS"""
        #normal = self._get_normal_from_terrain()

        # TODO: I WANT NORMAL FROM THE ROBOT BODY UP
        R_r = self.robot_body.get_matrix()[:3, :3]
        normal = R_r[:, 2]  # body up vector (world frame)
        normal /= np.maximum(np.linalg.norm(normal), 1e-8)

        leg_tilt_reward = 0
        if normal is not None:
            for index, (leg, tip) in enumerate(zip(self.robot_legs, self.robot_feet)):
                current_leg_pos = leg.get_position()
                current_leg_tip_pos = tip.get_position()
                direction_of_the_leg = current_leg_pos - current_leg_tip_pos
                direction_of_the_leg /= np.maximum(np.linalg.norm(direction_of_the_leg), 1e-8)
                result = np.dot(direction_of_the_leg, normal)
                result = np.clip(result, 0, 1)

                result = result - 1

                #leg_rewards.append(f"{result * proximities[index]}")

                leg_tilt_reward += 1/6 * (result * proximities[index])
                #leg_tilt_reward += 1/6 * result


        leg_rewards.append(f"{normal}")

        if goal_reached:
            goal_reached_reward = 1

        if not goal_reached and done:
            fell_to_the_side_reward = -1

        """ Penalize body Z - position"""
        #x, y, z = self.robot_body.get_position()
        z = self.robot_body.check_distance(self.terrain)
        #z = self.distance_sensor.read()
        #print(z)

        #print(f"z: {z} | {Object.get_object('hexa_body').check_distance(Object.get_object('Terrain_shape'))}")

        """ Proportional to the Z - position"""
        #print(z)
        # if z != -1.0 and z <= 0.05:
        #     body_height_reward = -1 + (z / 0.05)
        # else:
        #     body_height_reward = 0
        if z != -1.0:
            target_height = 0.25
            body_height_reward = -min(abs(z - target_height), 1.0)
        else:
            body_height_reward = 0

        # self.body_position_normalizer.update_single(z)
        # z = self.body_position_normalizer.normalize(z)[0]
        # body_height_reward = np.clip(z / 3.0, -1, 1)

        # _, omega = self.robot_body.get_velocity()
        # omega_xy = omega[:2]
        # omega_tilt = np.linalg.norm(omega_xy)
        # if warmup:
        #     self.angular_velocity_normalizer.update(omega_tilt)
        # omega_tilt = self.angular_velocity_normalizer.normalize_restricted(omega_tilt) ** 2
        # angular_velocity_reward = -1 * omega_tilt

        """ Reward for staying alive (encourages robot to not roll over)"""

        roll, pitch, _ = self.robot_body.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            fell_over = True


        match self.rewards:
            case 2:
                stability_reward = (
                        0.4 * tilt_reward +
                        0.6 * body_height_reward
                        #0.2 * angular_velocity_reward
                )
                legs_reward = (
                        0.9 * leg_tip_reward +
                        0.1 * leg_tilt_reward
                )
                motion_reward = (
                        1.0 * velocity_reward
                        #0.01 * correct_direction_reward
                )
                total_reward = (
                        2.0 * stability_reward +
                        1.0 * legs_reward +
                        1.0 * motion_reward +
                        0.01
                )

                total_reward = np.clip(total_reward, -4, 4)
                if fell_over:
                    total_reward -= 5

            case _:
                    total_reward = 0


        """ Scale to [-10; 10]"""
        #total_reward = np.clip(total_reward, -1, 1)

        return total_reward, leg_rewards


    # def _calculate_reward_her(self, state, robot_body_pos, goal_pos, done):
    #
    #     #observation = np.concatenate((target_direction, positions, joint_velocities, hexa_orient, linear, angular,
    #     #                              [tilt_cos], foot_contacts, [body_height]))
    #     # 3 18 18 6 3 3 1 6 1
    #
    #     SPECS = [
    #         ("target_direction", 3),
    #         ("positions", 18),
    #         ("joint_velocities", 18),
    #         ("hexa_orient", 6),
    #         ("linear", 3),
    #         ("angular", 3),
    #         ("tilt_cos", 1),
    #         ("foot_contacts", 6),
    #         ("body_height", 1),
    #     ]
    #
    #     splits = np.cumsum([size for _, size in SPECS])[:-1]
    #     parts = np.split(state, splits)
    #
    #     obs = {
    #         name: part if size > 1 else part[0]
    #         for (name, size), part in zip(SPECS, parts)
    #     }
    #
    #
    #
    #
    #     velocity = self._calculate_robot_velocity(goal_pos, robot_body_pos, obs["linear"])
    #     velocity_reward = self.velocity_normalizer.normalize(velocity)
    #
    #
    #     #roll, pitch, _ = self.robot_body.get_orientation()
    #     tilt_reward, goal_reached_reward, fell_to_the_side_reward, body_height_reward = 0, 0, 0, 0
    #
    #     fell_over = False
    #
    #     leg_tip_reward = 0
    #
    #     scale = 100.0
    #     proximities = []
    #     for i in range(6):
    #         d = obs["foot_contacts"][i]
    #         proximity = np.exp(-scale * d)
    #         proximities.append(proximity)
    #         leg_tip_reward += (1 / 6) * proximity
    #
    #     # TODO
    #     # tilt = self._get_robot_tilt()
    #     # """ To ensure -1; 1 range"""
    #     # tilt_reward = 2 * ((-1.0 + tilt) + 1) -1
    #
    #     # """ ROBOT LEGS"""
    #     # normal = self._get_normal_from_terrain()
    #     # leg_tilt_reward = 0
    #     # if normal is not None:
    #     #     for index, (leg, tip) in enumerate(zip(self.robot_legs, self.robot_feet)):
    #     #         current_leg_pos = leg.get_position()
    #     #         current_leg_tip_pos = tip.get_position()
    #     #         direction_of_the_leg = current_leg_pos - current_leg_tip_pos
    #     #         direction_of_the_leg /= np.maximum(np.linalg.norm(direction_of_the_leg), 1e-8)
    #     #         result = np.dot(direction_of_the_leg, normal)
    #     #         result = np.clip(result, 0, 1)
    #     #
    #     #         leg_tilt_reward += 1/6 * (result * proximities[index])
    #
    #
    #     # if robot_body_pos:
    #     #     goal_reached_reward = 1
    #
    #     """ Penalize body Z - position"""
    #     #x, y, z = self.robot_body.get_position()
    #     # TODO
    #     #z = self.robot_body.check_distance(self.terrain)
    #     #z = self.distance_sensor.read()
    #     #print(z)
    #
    #     """ Proportional to the Z - position"""
    #     #print(z)
    #     # if z != -1.0 and z <= 0.10:
    #     #     body_height_reward = -1 + (z / 0.10)
    #     # else:
    #     #     body_height_reward = 0.001
    #
    #
    #     # _, omega = self.robot_body.get_velocity()
    #     # omega_xy = omega[:2]
    #     # omega_tilt = np.linalg.norm(omega_xy)
    #
    #
    #     #omega_tilt = self.angular_velocity_normalizer.normalize_restricted(omega_tilt) ** 2
    #     #angular_velocity_reward = -1 * omega_tilt
    #
    #     """ Reward for staying alive (encourages robot to not roll over)"""
    #
    #     # roll, pitch, _ = self.robot_body.get_orientation()
    #     #
    #     # if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
    #     #     fell_over = True
    #
    #
    #     match self.rewards:
    #         case 2:
    #             if done:
    #                 total_reward = -1
    #             else:
    #                 # stability_reward = (
    #                 #         0.5 * tilt_reward +
    #                 #         0.5 * body_height_reward
    #                 #         #0.2 * angular_velocity_reward
    #                 # )
    #                 # legs_reward = (
    #                 #         0.3 * leg_tip_reward +
    #                 #         0.7 * leg_tilt_reward
    #                 # )
    #                 motion_reward = (
    #                         1.0 * velocity_reward
    #                 )
    #                 total_reward = (
    #                         # 0.05 * stability_reward +
    #                         # 0.05 * legs_reward +
    #                         2.0 * motion_reward# +
    #                         #goal_reached_reward
    #                 )
    #
    #                 total_reward = np.clip(total_reward, -4, 4)
    #         case _:
    #             total_reward = 0
    #
    #     return total_reward
    """ Method that checks if the episode should end """
    def _check_termination(self):
        roll, pitch, _ = self.robot_body.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            #print("Too much tilt")
            return True, False

        """ Terminate episode if the robot is close enough to the target"""
        #current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        if Object.get_object("hexapod").check_collision(self.goal):
            self.results_queue.put("Target reached collision")
            return False, True

        if Object.get_object("hexapod").check_distance(self.goal) < 1.0:
            self.results_queue.put("Target reached distance")
            return False, True

        return False, False

    """ Handling PyRep simulation termination"""
    def close(self):
        self.pr.stop()
        self.pr.shutdown()
