import ctypes

import numpy as np
from gym import spaces
from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from cffi import FFI

from DDPG.running_min_max_normalizer import RunningMinMaxNormalizer
from DDPG.running_normalizer import RunningNormalizer


""" Class responsible for defining environment rules and simulation logic """
class HexapodEnv:
    def __init__(self, disable_rendering=False, rewards=1):
        """ Initialize PyRep connection """
        self.pr = PyRep()
        self.pr.launch('red_box_scene_v3.ttt', headless=disable_rendering)
        self.pr.start()

        """ Load all joints of hexapod """
        self.joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
        self.joints = [Joint(name) for name in self.joint_names]

        """ Define action and observation space """
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(135,), dtype=np.float32)

        self.vision_sensor = VisionSensor('Photo_sensor')

        self.previous_distance = None

        self.initial_distance = 0

        self.previous_joint_positions = None

        self.initial_observation = None

        self.terminate = False

        self.velocity_towards_goal = None

        self.rewards = rewards

        self.robot = Object.get_object('hexa_base')
        self.robot_body = Object.get_object('hexa_body')
        self.goal = Object.get_object('Target')

        self.initial_position = self.robot.get_position()
        self.initial_orientation = self.robot.get_orientation()

        self.max_tilt_penalization = np.deg2rad(75)
        self.max_tilt_termination = np.deg2rad(90)

        #self.reward_normalizer = RunningNormalizer(shape=(1,))
        self.target_range = 2000.0 / 300.0

        self.distance_normalizer = RunningNormalizer(shape=(1,))
        #self.velocity_normalizer = RunningNormalizer(shape=(1,))
        self.velocity_normalizer = RunningMinMaxNormalizer()

        self.reward_min = float('inf')
        self.reward_max = float('-inf')

        self.current_distances = []
        self.current_velocities = []
        self.current_rewards = []

        # TODO 10 ?
        self.max_simulation_count = 5


    """ Method responsible for setting new joint positions, advancing to the next state and returning relevant information about this action"""
    def set_joint_positions(self, action, episode):
        for joint, value in zip(self.joints, action):
            joint.set_joint_target_position(value)

    def step_and_update(self):
        self.pr.step()
        self._update_state()

    def get_step_results(self, episode, warmup = False):
        obs = self._get_observation()
        done, goal_reached = self._check_termination()
        reward = self._calculate_reward(episode, goal_reached, done, warmup)

        return obs, reward, done, goal_reached

    def all_joints_reached_targets(self, position_tolerance=0.05, velocity_tolerance=0.05):
        for index, j in enumerate(self.joints):
            pos = j.get_joint_position()
            target = j.get_joint_target_position()
            vel_lin, vel_ang = j.get_velocity()

            pos_not_synced = abs(pos - target) > position_tolerance

            high_velocity = np.linalg.norm(vel_lin) > velocity_tolerance and np.linalg.norm(vel_ang) > velocity_tolerance

            if pos_not_synced and high_velocity:
                return False

        return True

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
        observation = self._get_observation()

        # print(observation)
        # print("-------------")

        return observation

    def _update_state(self):
        self.robot = Object.get_object('hexa_base')
        self.robot_body = Object.get_object('hexa_body')

    """ Method responsible for handling observation space logic """
    def _get_observation(self):
        """ Get hexapod joint positions """
        num_joints = len(self.joints)
        linear_velocities = np.zeros(num_joints*3)
        angular_velocities = np.zeros(num_joints*3)
        positions = np.zeros(num_joints)

        """ Get both linear and angular joint velocities """
        for i, joint in enumerate(self.joints):
            linear, angular = joint.get_velocity()

            linear_velocities[i * 3:(i + 1) * 3] = linear
            angular_velocities[i * 3:(i + 1) * 3] = angular
            positions[i] = joint.get_joint_position()


        """ Get relevant information about hexapod """
        linear, angular = self.robot.get_velocity()
        hexa_orient = self.robot.get_orientation()

        observation = np.concatenate((positions, linear_velocities, angular_velocities, hexa_orient, linear, angular))

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


        roll, pitch, _ = self.robot.get_orientation()
        tilt_reward, goal_reached_reward, fell_to_the_side_reward, body_height_reward = 0, 0, 0, 0
        alive_reward = 1

        fell_over = False
        """ Penalize too much tilt"""
        if abs(roll) >= self.max_tilt_penalization:
            """ Scale to range [-1; -0.7]"""
            tilt_reward = -(min(abs(roll), np.deg2rad(90.0)) / self.max_tilt_termination)
        elif abs(pitch) >= self.max_tilt_penalization:
            tilt_reward = -(min(abs(pitch), np.deg2rad(90.0)) / self.max_tilt_termination)
        else:
            tilt_reward = 0.001

        if tilt_reward == -1:
            #print("Robot fell")
            fell_over = True



        if goal_reached:
            goal_reached_reward = 1

        if not goal_reached and done:
            fell_to_the_side_reward = -1

        """ Penalize body Z - position"""
        x, y, z = self.robot_body.get_position()

        """ Proportional to the Z - position"""
        #print(z)
        if z <= 0.2:
            body_height_reward = -1 + (z / 0.20)
        else:
            body_height_reward = 0.001

        # self.body_position_normalizer.update_single(z)
        # z = self.body_position_normalizer.normalize(z)[0]
        # body_height_reward = np.clip(z / 3.0, -1, 1)

        """ Reward for staying alive (encourages robot to not roll over)"""

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
                elif goal_reached:
                    total_reward = 10
                else:
                    total_reward = (
                        0.5 * (velocity_reward)+
                        0.5 * (tilt_reward + body_height_reward) / 2
                    )
                    total_reward = np.clip(total_reward, -1, 1)
            case _:
                total_reward = 0


        """ Scale to [-10; 10]"""
        #total_reward = np.clip(total_reward, -1, 1)

        return total_reward

    """ Method that checks if the episode should end """
    def _check_termination(self):
        roll, pitch, _ = self.robot.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            #print("Too much tilt")
            return True, False

        """ Terminate episode if the robot is close enough to the target"""
        current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        if current_distance < 1.0:
            print("Target reached")
            return True, True

        return False, False

    """ Handling PyRep simulation termination"""
    def close(self):
        self.pr.stop()
        self.pr.shutdown()
