import ctypes

import numpy as np
from gym import spaces
from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from cffi import FFI

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

        self.robot = Object.get_object('hexa_body')
        self.goal = Object.get_object('Target')

        self.initial_position = self.robot.get_position()
        self.initial_orientation = self.robot.get_orientation()

        self.max_tilt_penalization = np.deg2rad(75)
        self.max_tilt_termination = np.deg2rad(90)

        self.reward_normalizer = RunningNormalizer(shape=(1,))
        self.target_range = 2000.0 / 300.0



    """ Method responsible for setting new joint positions, advancing to the next state and returning relevant information about this action"""
    def step(self, action, episode):
        for joint, value in zip(self.joints, action):
            joint.set_joint_target_position(value)

        self.pr.step()

        self._update_state()

        obs = self._get_observation()
        done, goal_reached = self._check_termination()
        reward = self._calculate_reward(episode, goal_reached, done)

        return obs, reward, done, goal_reached

    """ Method responsible for resetting the simulation"""
    def reset(self):
        self.previous_distance = None
        self.previous_joint_positions = None

        self.pr.stop()

        self.pr.start()
        observation = self._get_observation()

        # print(observation)
        # print("-------------")

        return observation

    def _update_state(self):
        self.robot = Object.get_object('hexa_body')

    """ Method responsible for handling observation space logic """
    def _get_observation(self):
        """ Get hexapod joint positions """
        positions = np.array([joint.get_joint_position() for joint in self.joints])
        linear_velocities = []
        angular_velocities = []

        """ Get both linear and angular joint velocities """
        for joint in self.joints:
            linear, angular = joint.get_velocity()
            linear_velocities.append(linear)
            angular_velocities.append(angular)

        linear_velocities = np.array(linear_velocities).flatten()
        angular_velocities = np.array(angular_velocities).flatten()

        """ Get relevant information about hexapod """
        linear, angular = self.robot.get_velocity()
        hexa_orient = self.robot.get_orientation()

        observation = np.concatenate((positions, linear_velocities, angular_velocities, hexa_orient, linear, angular))

        return observation.astype(np.float32)

    """ Method responsible for calculating velocity of hexapod towards the goal"""
    def _calculate_robot_velocity(self):
        direction = self.goal.get_position().flatten() - self.robot.get_position().flatten()
        direction_norm = np.linalg.norm(direction)
        direction_to_goal_normalized = direction / (direction_norm + 1e-8)

        linear, _ = self.robot.get_velocity()
        velocity = np.array(linear).flatten()
        velocity_toward_goal = float(np.dot(velocity, direction_to_goal_normalized))

        return velocity_toward_goal

    """ Method responsible for calculating reward based on the current state of hexapod """
    def _calculate_reward(self, episode, goal_reached, done):

        current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())

        """ Save previous distance to target"""
        if self.previous_distance is None:
            #self.initial_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
            #self.previous_distance = self.initial_distance
            distance_reward = 0

        else:

            """ Calculate reward, based on distance difference """
            distance_reward = (self.previous_distance - current_distance)

        self.previous_distance = current_distance

        velocity_reward = self._calculate_robot_velocity()



        roll, pitch, _ = self.robot.get_orientation()
        tilt_reward, goal_reached_reward, fell_to_the_side_reward = 0, 0, 0
        """ Penalize too much tilt"""
        if abs(roll) > self.max_tilt_penalization or abs(pitch) > self.max_tilt_penalization:
            tilt_reward = -10

        if goal_reached:
            goal_reached_reward = 500

        if not goal_reached and done:
            fell_to_the_side_reward = -100

        x, y, z = self.robot.get_position()

        body_height_reward = abs(z) * 10

        """ Reward for staying alive (encourages robot to not roll over)"""
        #alive_reward = 1

        match self.rewards:
            case 1:
                total_reward = (distance_reward *100) + (velocity_reward * 10) + tilt_reward + goal_reached_reward + fell_to_the_side_reward# + alive_reward
            case 2:
                total_reward = (distance_reward * 100) + (velocity_reward * 10) + tilt_reward + goal_reached_reward + fell_to_the_side_reward + body_height_reward

        #self.reward_normalizer.update(np.array([[total_reward]], dtype=np.float32))

        # if episode > 100:
        #     normalized_reward = self.reward_normalizer.normalize(np.array([[total_reward]], dtype=np.float32))[0, 0]
        #
        #     # Scale to your desired target range (e.g., ±2000)
        #     scaled_reward = normalized_reward * self.target_range
        #
        #     #return total_reward
        #     print(f"{total_reward} vs {normalized_reward:.2f} vs {scaled_reward:.2f}")
        #
        #     return scaled_reward

        return round(total_reward, 5)

    """ Method that checks if the episode should end """
    def _check_termination(self):
        roll, pitch, _ = self.robot.get_orientation()

        if abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination:
            #print("Too much tilt")
            return True, False

        """ Terminate episode if the robot is close enough to the target"""
        current_distance = np.linalg.norm(self.robot.get_position() - self.goal.get_position())
        if current_distance < 3.0:
            print("Target reached")
            return True, True

        return False, False

    """ Handling PyRep simulation termination"""
    def close(self):
        self.pr.stop()
        self.pr.shutdown()
