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
    def __init__(self, disable_rendering=False, rewards=1):
        """ Initialize PyRep connection """
        self.pr = PyRep()
        self.pr.launch('red_box_scene_v5.ttt', headless=disable_rendering)
        self.pr.start()

        """ Load all joints of hexapod """
        self.joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
        self.joints = [Joint(name) for name in self.joint_names]

        """ Define action and observation space """
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(60,), dtype=np.float32)

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
            self.robot_feet.append(Object.get_object(f"hexa_footTarget{i}"))
            self.robot_legs.append(Object.get_object(f"hexa_link3_{i}"))

        self.max_tilt_penalization = np.deg2rad(75)
        self.max_tilt_termination = np.deg2rad(90)

        #self.reward_normalizer = RunningNormalizer(shape=(1,))
        self.target_range = 2000.0 / 300.0

        self.distance_normalizer = RunningNormalizer(shape=(1,))
        #self.velocity_normalizer = RunningNormalizer(shape=(1,))
        self.velocity_normalizer = RunningMinMaxNormalizer()
        self.angular_velocity_normalizer = RunningMinMaxNormalizer()

        self.reward_min = float('inf')
        self.reward_max = float('-inf')

        self.current_distances = []
        self.current_velocities = []
        self.current_rewards = []

        # TODO 10 ?
        self.max_simulation_count = 5
        self.max_simulation_count_reset = 50

        self.T1 = 25_000  # stabilita → nohy
        self.T2 = 50_000  # nohy → pohyb
        self.T3 = 100_000  # pohyb → cíl

        self.phase_limit = 25_000
        self.current_phase = 1
        self.max_phase_num = 3

        self.w_stability = 1.0

        self.w_legs = 0

        self.w_motion = 0

        self.w_goal = 0

        self.current_global_step_count = 0


    """ Method responsible for setting new joint positions, advancing to the next state and returning relevant information about this action"""
    def set_joint_positions(self, action, episode):
        for joint, value in zip(self.joints, action):
            joint.set_joint_target_position(value)

    def step_and_update(self):
        self.pr.step()
        #self._update_state()

    def get_step_results(self, episode, warmup = False):
        obs = self._get_observation()
        done, goal_reached = self._check_termination()
        reward = self._calculate_reward(episode, goal_reached, done, warmup)

        return obs, reward, done, goal_reached

    def all_joints_reached_targets(self, position_tolerance=0.05, velocity_tolerance=0.05):
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

    def _randomly_place_robot_target(self):
        #robot = Object.get_object("hexapod")
        robot_position = self.robot.get_position()
        target_position = self.goal.get_position()

        # base = Object.get_object("Terrain_shape")
        # base_dimensions = base.get_bounding_box()
        # wall = Object.get_object("Walls")


        def _move_object_to_new_position(affected_object, z_position, collision_object):
            while True:
                new_x, new_y = self._generate_new_position(self.base_dimensions[0], self.base_dimensions[1], self.base_dimensions[2], self.base_dimensions[3], 3.0)
                affected_object.set_position([new_x, new_y, z_position])
                self.pr.step()
                if not affected_object.check_collision(collision_object):
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

        """ Keep the original Z-coordinate, set new coordinates """

        _move_object_to_new_position(self.robot, robot_position[2], self.wall)
        _move_object_to_new_position(self.goal, robot_position[2], self.wall)

        """ Ensure that robot and goal are not position right next to each other"""
        _ensure_robot_goal_distance()

        """ Wait until the robot and target reaches ground"""
        current_simulation_count = 0
        last_cube_pos, last_robot_pos = float("inf"), float("inf")
        current_cube_pos, current_robot_pos = self.goal.check_distance(self.terrain), self.robot.check_distance(self.terrain)
        while current_cube_pos < last_cube_pos or current_robot_pos < last_robot_pos:
            self.pr.step()
            current_simulation_count += 1
            #print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))

            if current_simulation_count >= self.max_simulation_count_reset:
                print("Maximum simulation count reached")
                print(self.robot.get_position())
                print(self.distance_sensor.read())
                print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))
                print(self.goal.get_position())
                # print(robot.check_distance(Object.get_object("Terrain_shape")))
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
                print("Maximum simulation count reached reverb")
                print(self.robot.get_position())
                print(self.distance_sensor.read())
                print(Object.get_object("hexapod").check_distance(Object.get_object("Terrain_shape")))
                print(self.goal.get_position())
                break

            last_robot_pos = current_robot_pos
            current_robot_pos = self.robot.check_distance(self.terrain)


    """ Method responsible for resetting the simulation"""
    def reset(self):
        self.previous_distance = None
        self.previous_joint_positions = None

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
        self._randomly_place_robot_target()

        observation = self._get_observation()

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
    def _get_observation(self):
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
        robot_pos = self.robot.get_position()
        goal_pos = self.goal.get_position()
        distance_from_target = np.linalg.norm(goal_pos - robot_pos)

        roll, pitch, yaw = self.robot_body.get_orientation()
        hexa_orient = np.array([
            np.sin(roll), np.cos(roll),
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw)
        ])

        tilt_cos = self._get_robot_tilt()

        linear, angular = self.robot.get_velocity()
        hexa_position = self.robot.get_position()#self.robot.get_position()
        target_direction = self.goal.get_position() - hexa_position
        target_direction = target_direction / np.linalg.norm(target_direction)
        foot_contacts = np.zeros(6)
        for i in range(6):
            foot_contacts[i] = self.robot_feet[i].check_distance(self.terrain)

        body_height = self.robot_body.check_distance(self.terrain)

        observation = np.concatenate((positions, joint_velocities, hexa_orient, linear, angular, [distance_from_target], target_direction, [tilt_cos], foot_contacts, [body_height]))

        return observation.astype(np.float32)

    """ Method responsible for calculating velocity of hexapod towards the goal"""
    def _calculate_robot_velocity(self):
        direction = self.goal.get_position() - self.robot.get_position()
        direction_norm = np.linalg.norm(direction)
        direction_to_goal_normalized = direction / (direction_norm + 1e-8)

        linear, _ = self.robot.get_velocity()

        velocity_toward_goal = float(np.dot(linear.flatten(), direction_to_goal_normalized))

        return velocity_toward_goal

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
            n /= np.linalg.norm(n)
            return n

        return None

    def _get_robot_tilt(self):
        n = self._get_normal_from_terrain()
        if n is not None:
            R_r = self.robot_body.get_matrix()[:3, :3]
            z = R_r[:, 2]
            z /= np.linalg.norm(z)

            cos_theta = np.dot(z, n)
            return max(0, cos_theta)

        return 0

    def _smoothstep(self, x):
        return x * x * (-2 * x + 3)

    """ Method responsible for calculating reward based on the current state of hexapod """
    def _calculate_reward(self, episode, goal_reached, done, warmup):
        current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        distance = 0

        if self.previous_distance is not None:
            distance = self.previous_distance - current_distance

        if distance != 0:
            self.current_distances.append(distance)
            #self.distance_normalizer.update(distance)
        distance_reward = self.distance_normalizer.normalize(distance)[0]
        distance_reward = np.clip(distance_reward, -3, 3)


        self.previous_distance = current_distance


        velocity = self._calculate_robot_velocity()
        if warmup:
            self.velocity_normalizer.update(velocity)
        velocity_reward = self.velocity_normalizer.normalize(velocity)
        #velocity_reward = np.clip(velocity_reward, -3, 3)


        """ Scale to range [-1; 1] """
        #velocity_reward = np.clip(velocity_reward, -1, 1)
        #print("velocity_reward ", velocity_reward)


        roll, pitch, _ = self.robot_body.get_orientation()
        tilt_reward, goal_reached_reward, fell_to_the_side_reward, body_height_reward = 0, 0, 0, 0
        alive_reward = 1

        fell_over = False

        leg_tip_reward = 0
        # for i in range(6):
        #     if self.robot_feet[i].check_distance(self.terrain) < 0.01:
        #         leg_tip_reward += 1/6

        scale = 100.0
        proximities = []
        for i in range(6):
            d = self.robot_feet[i].check_distance(self.terrain)
            proximity = np.exp(-scale * d)
            proximities.append(proximity)
            leg_tip_reward += (1 / 6) * proximity



        tilt = self._get_robot_tilt()
        """ To ensure -1; 1 range"""
        tilt_reward = 2 * ((-1.0 + tilt) + 1) -1


        """ ROBOT LEGS"""
        normal = self._get_normal_from_terrain()
        leg_tilt_reward = 0
        if normal is not None:
            for index, (leg, tip) in enumerate(zip(self.robot_legs, self.robot_feet)):
                current_leg_pos = leg.get_position()
                current_leg_tip_pos = tip.get_position()
                direction_of_the_leg = current_leg_pos - current_leg_tip_pos
                direction_of_the_leg /= np.linalg.norm(direction_of_the_leg)
                result = np.dot(direction_of_the_leg, normal)
                result = np.clip(result, 0, 1)

                leg_tilt_reward += 1/6 * (result * proximities[index])


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
        if z != -1.0 and z <= 0.10:
            body_height_reward = -1 + (z / 0.10)
        else:
            body_height_reward = 0.001

        # self.body_position_normalizer.update_single(z)
        # z = self.body_position_normalizer.normalize(z)[0]
        # body_height_reward = np.clip(z / 3.0, -1, 1)

        _, omega = self.robot_body.get_velocity()
        omega_xy = omega[:2]
        omega_tilt = np.linalg.norm(omega_xy)
        if warmup:
            self.angular_velocity_normalizer.update(omega_tilt)
        omega_tilt = self.angular_velocity_normalizer.normalize_restricted(omega_tilt) ** 2
        angular_velocity_reward = -1 * omega_tilt

        """ Reward for staying alive (encourages robot to not roll over)"""

        roll, pitch, _ = self.robot_body.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            fell_over = True


        match self.rewards:
            # case 1:
            #     total_reward = (
            #         0.6 * (distance_reward + velocity_reward + tilt_reward) / 3 +
            #         0.3 * (goal_reached_reward) +
            #         0.1 * (fell_to_the_side_reward)
            #     )
                #total_reward = (distance_reward *100) + (velocity_reward * 10) + tilt_reward + goal_reached_reward + fell_to_the_side_reward# + alive_reward
            # case 2:
            #     total_reward = (
            #             np.clip( 0.3 * ( velocity_reward ), -1.0, 1.0) +
            #             0.7 * (tilt_reward + body_height_reward)/2
            #             )
            case 2:
                if fell_over:
                    total_reward = -1
                # # elif goal_reached:
                # #     total_reward = 10
                else:
                    # total_reward = (
                    #         0.1 * (tilt_reward) +
                    #         0.2 * (body_height_reward) +
                    #         0.1 * angular_velocity_reward +
                    #         0.3 * leg_tip_reward +
                    #         0.3 * leg_tilt_reward
                    # )
                    stability_reward = (
                            0.3 * tilt_reward +
                            0.5 * body_height_reward +
                            0.2 * angular_velocity_reward
                    )
                    legs_reward = (
                            0.3 * leg_tip_reward +
                            0.7 * leg_tilt_reward
                    )
                    motion_reward = (
                        1.0 * velocity_reward
                    )

                    # total_reward = (
                    #         self.w_stability * stability_reward +
                    #         self.w_legs * legs_reward +
                    #         self.w_motion * motion_reward +
                    #         self.w_goal * 1.0
                    # )
                    if self.current_phase == 1:
                        stability_reward = (
                                0.3 * tilt_reward +
                                0.5 * body_height_reward +
                                0.2 * angular_velocity_reward
                        )
                        legs_reward = (
                                0.3 * leg_tip_reward +
                                0.7 * leg_tilt_reward
                        )
                        motion_reward = (
                                1.0 * velocity_reward
                        )

                        total_reward = (
                            stability_reward
                        )
                    elif self.current_phase == 2:
                        stability_reward = (
                                0.3 * tilt_reward +
                                0.5 * body_height_reward +
                                0.2 * angular_velocity_reward
                        )
                        legs_reward = (
                                0.3 * leg_tip_reward +
                                0.7 * leg_tilt_reward
                        )
                        motion_reward = (
                                1.0 * velocity_reward
                        )

                        total_reward = (
                            stability_reward +
                            legs_reward
                        )
                    else:
                        stability_reward = (
                                0.5 * tilt_reward +
                                0.5 * body_height_reward
                                #0.2 * angular_velocity_reward
                        )
                        legs_reward = (
                                0.3 * leg_tip_reward +
                                0.7 * leg_tilt_reward
                        )
                        motion_reward = (
                                1.0 * velocity_reward
                        )
                        total_reward = (
                                0.10 * stability_reward +
                                0.05 * legs_reward +
                                2.0 * motion_reward +
                                goal_reached_reward
                        )

                    total_reward = np.clip(total_reward, -4, 4)
            case _:
                total_reward = 0


        """ Scale to [-10; 10]"""
        #total_reward = np.clip(total_reward, -1, 1)

        return total_reward

    """ Method that checks if the episode should end """
    def _check_termination(self):
        roll, pitch, _ = self.robot_body.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            #print("Too much tilt")
            return True, False

        """ Terminate episode if the robot is close enough to the target"""
        #current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        if Object.get_object("hexapod").check_collision(self.goal):
            print("Target reached collision")
            return False, True

        if Object.get_object("hexapod").check_distance(self.goal) < 1.0:
            print("Target reached distance")
            return False, True

        return False, False

    """ Handling PyRep simulation termination"""
    def close(self):
        self.pr.stop()
        self.pr.shutdown()
