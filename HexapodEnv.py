import numpy as np
import gym
from gym import spaces
from pyrep import PyRep
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object


class HexapodEnv():#gym.Env):
    def __init__(self):
        #super(HexapodEnv, self).__init__()

        # Initialize PyRep & load the scene
        self.pr = PyRep()
        #self.pr.set_simulation_timestep(0.05)
        self.pr.launch('seminarka_scena.ttt', headless=True)
        #self.pr.set_simulation_timestep(1.00)
        # self.pr.step_ui()
        # print(self.pr.get_simulation_timestep())
        self.pr.start()

        # Get joints
        self.joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
        self.joints = [Joint(name) for name in self.joint_names]

        # Continuous action space: 18 joints each with a value between -0.5 and 0.5
        # Generuje mi to rozsah
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(18,), dtype=np.float32)
        #self.high = 140
        #self.low = -140

        # Observation space (joint positions + velocity)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(36,), dtype=np.float32)

        self.previous_distance = 0

        self.initial_distance = 0

        self.previous_joint_positions = None

        self.initial_observation = None

        self.terminate = False

    def step(self, action):
        """Apply the action and update the simulation."""
        if len(action) != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} actions, got {len(action)}")

        # Apply action (joint angles)
        #print(action)
        index = 0

        for joint, value in zip(self.joints, action):
            #if value < -1.5 or value > 1.5:
                #return "Error"
            #print(value)
            #joint.set_joint_force(value)
            #joint.set_joint_target_velocity(0.5)
            #joint.set_joint_position(value, True)
            joint.set_joint_target_position(value)
            #joint.set_joint_target_position(np.deg2rad(-100.0))
            #joint.set_joint_target_position(joint.get_joint_target_position()+0.01)
            #index += 1


            # if 12 <= index <= 16:
            #     joint.set_joint_target_position(joint.get_joint_target_position()+0.05)
            # index += 1


        #print(np.array([(joint.get_joint_target_position()) for joint in self.joints]))

        self.pr.step()  # Advance simulation

        obs = self._get_observation()#np.array([(joint.get_joint_position()) for joint in self.joints])
        reward = self._calculate_reward()
        done = self._check_termination()

        return obs, reward, done, {}

    def reset(self):
        """Reset the environment and return the initial observation."""
        #print(np.array([(joint.get_joint_target_position()) for joint in self.joints]))

        #print("Last:", self._get_observation())
        # TODO: Observaci bereme drive nez zastavime prostredi
        #observation = self._get_observation()
        self.pr.stop()
        self.previous_distance = 0
        self.previous_joint_positions = None
        self.pr.start()
        #observation = self._get_observation()
        #print(observation)

        #observation = self._get_observation()

        # Reset joints
        for joint in self.joints:
            joint.set_joint_position(0.0)

        observation = self._get_observation()

        #self.pr.step()

        return observation

    def _get_observation(self):
        """Get sensor/joint data for RL."""
        #return np.array([(joint.get_joint_position()) for joint in self.joints])#self.initial_observation

        positions = np.array([joint.get_joint_position() for joint in self.joints])
        velocities = np.array([joint.get_joint_velocity() for joint in self.joints])
        #print(velocities)

        return np.concatenate((positions, velocities))
    def _calculate_reward(self):
        """Reward function to encourage forward movement."""
        robot = Object.get_object('hexa_base')
        goal = Object.get_object('Cuboid')

        # Inicializace počáteční vzdálenosti
        if self.initial_distance == 0:
            self.initial_distance = np.linalg.norm(robot.get_position() - goal.get_position())

        current_distance = np.linalg.norm(robot.get_position() - goal.get_position())

        # Odměna za snížení vzdálenosti
        distance_reward = (self.previous_distance - current_distance)
        if distance_reward < 0:
            distance_reward *= 2
        else:
            distance_reward *= 10
        self.previous_distance = current_distance

        # Odměna za pohyb
        movement_reward = 0
        if self.previous_joint_positions is None:
            self.previous_joint_positions = np.array([(joint.get_joint_position()) for joint in self.joints])
        else:
            for index, joint in enumerate(self.joints):
                #print(joint.get_joint_position())
                #print(self.previous_joint_positions[index])
                if round(joint.get_joint_position(),2) == round(self.previous_joint_positions[index], 2):
                    movement_reward -= 0.01
                    #print("NO MOVEMENT")
                else:
                    movement_reward += 0.01
            self.previous_joint_positions = np.array([(joint.get_joint_position()) for joint in self.joints])
            #if movement_reward < 0.04:
                #self.terminate = True


        # TODO: TEST - Odměna za směr
        direction = goal.get_position().flatten() - robot.get_position().flatten()
        direction_norm = np.linalg.norm(direction)
        direction_to_goal_normalized = direction / (direction_norm + 1e-8)

        # 2. Rychlost robota (vektor [vx, vy, vz]) + flatten pro prevod (1,3) na (3,)

        linear, _ = robot.get_velocity()

        velocity = np.array(linear).flatten()
        #print("Velocity:", velocity)

        #print("Velocity shape:", velocity.shape)
        #print("Direction shape:", direction_to_goal_normalized)

        # 3. Projekce rychlosti do směru k cíli (skalární součin)
        velocity_toward_goal = float(np.dot(velocity, direction_to_goal_normalized))

        #print("Velocity_toward_goal:", velocity_toward_goal)

        #print("Vel shape:", velocity_toward_goal)

        # 4. Odměna za rychlost směrem k cíli
        velocity_reward = velocity_toward_goal/2

        #print(distance_reward)

        #TODO
        # Penalizace za nestabilitu (náklon robota)
        # orientation = robot.get_orientation()
        # tilt_penalty = abs(orientation[0]) + abs(orientation[1])

        if movement_reward < 0:
            total_reward = distance_reward + velocity_reward#+ (0.5 / current_distance)# + min(movement_reward,0.05)
        else:
            total_reward = distance_reward + velocity_reward# + (0.5/current_distance)# + max(movement_reward,0.05)

        return total_reward#-(current_distance/2)
        #return -(current_distance**2)

        # if self.initial_distance == 0:
        #     self.initial_distance = np.linalg.norm(robot.get_position() - goal.get_position())
        #     print(self.initial_distance)
        # if self.previous_joint_positions.size == 0:
        #     self.previous_joint_positions = self._get_observation()
        # else:
        # # Check if the robot changed is still on the same position (same joint position)
        #     if np.array_equal(self.previous_joint_positions, self._get_observation()):
        #         print("Not moved")
        #         return -50
        #
        # current_distance = np.linalg.norm(robot.get_position() - goal.get_position())

        # If we are on the right side
        # if self.initial_distance > current_distance:
        #     return 1 / current_distance
        # else:
        #     return -(2 / current_distance)
        #return -(2 / current_distance)

        #print(robot.get_position())

        # if self.previous_distance > 0:
        #     # We moved in the right direction
        #     if self.previous_distance > current_distance:
        #         self.previous_distance = current_distance
        #         return 5/current_distance
        # self.previous_distance = current_distance
        # return -(1/current_distance)

        #reward_touch_platform = 0
        #if robot.get_position()[2] >= 0.5:
        #    reward_touch_platform += 0.1

        #return (1/reward_dist) + reward_touch_platform

        #return 1 / current_distance

    def _check_termination(self):
        """Check if the episode should end (e.g., if the robot falls over)."""
        if self.terminate:
            self.terminate = False
            return True
        return False
    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def render(self, mode='human'):
        pass  # Ignore mode for now; PyRep handles visualization


# import numpy as np
# import gym
# from gym import spaces
# from pyrep import PyRep
# from pyrep.objects.joint import Joint
# from pyrep.objects.object import Object
#
#
# class HexapodEnv(gym.Env):
#     def __init__(self):
#         super(HexapodEnv, self).__init__()
#
#         # Initialize PyRep & load the scene
#         self.pr = PyRep()
#         self.pr.launch('seminarka_scena.ttt', headless=False)
#         self.pr.start()
#
#         # Get joints
#         self.joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
#         self.joints = [Joint(name) for name in self.joint_names]
#
#         # 6 legs × 3 joints per leg = 18 joints
#         # Each joint has 3 possible movements (-, 0, +)
#         self.action_space = spaces.Discrete(3 ** 6)  # Control groups of joints together
#
#         self.observation_space = spaces.Box(low=-1, high=1, shape=(18,), dtype=np.float32)  # 18 joint angle
#
#     def step(self, action):
#         movement_options = [-0.5, 0.0, 0.5]  # Discrete joint angles
#         action_list = [(action // (3 ** i)) % 3 for i in range(6)]  # Decode action
#
#         for i in range(6):  # Apply action to 6 joint groups
#             for j in range(3):  # Hip, knee, ankle
#                 self.joints[i * 3 + j].set_joint_target_position(movement_options[action_list[i]])
#
#         self.pr.step()  # Advance simulation
#
#         obs = self._get_observation()
#         reward = self._calculate_reward()
#         done = self._check_termination()
#
#         return obs, reward, done, {}
#
#     def reset(self):
#         """Reset the environment and return the initial observation."""
#         self.pr.stop()
#         self.pr.start()
#         return self._get_observation()
#
#     def _get_observation(self):
#         """Get sensor/joint data for RL."""
#         joint_angles = np.array([joint.get_joint_position() for joint in self.joints])
#         return np.clip(joint_angles, -1, 1)  # Normalize
#
#     def _calculate_reward(self):
#         """Reward function to encourage forward movement."""
#         robot = Object.get_object('hexa_base')
#         return robot.get_position()[0]  # Reward forward movement on x-axis
#
#     def _check_termination(self):
#         """Check if the episode should end (e.g., if the robot falls over)."""
#         robot = Object.get_object('hexa_base')
#         return abs(robot.get_orientation()[0]) > 0.6  # Too much tilt?
#
#     def close(self):
#         self.pr.stop()
#         self.pr.shutdown()