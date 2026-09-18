import math
import random
from queue import Full
import numpy as np
from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.backend import sim
from TD3.my_proximity_sensor import MyProximitySensor
from TD3.running_min_max_normalizer import RunningMinMaxNormalizer

class HexapodEnv:
    def __init__(self, results_queue, disable_rendering=False, experiment_num=0):
        """
        Initialization of class that is responsible for defining environment rules and simulation logic
        :param results_queue: results queue in which each worker inserts current episode performance
        :param disable_rendering: boolean flag controlling PyRep's headless mode
        :param experiment_num: chosen experiment number
        """
        self.results_queue = results_queue
        self.experiment_num = experiment_num
        self.disable_rendering = disable_rendering

        """ Initialize PyRep connection """
        self.pr = PyRep()
        if self.experiment_num == 1:
            self.pr.launch('red_box_scene_experiment1.ttt', headless=disable_rendering)
        elif self.experiment_num == 2:
            self.pr.launch('red_box_scene_experiment2.ttt', headless=disable_rendering)
        elif self.experiment_num == 3:
            self.pr.launch('red_box_scene_experiment3.ttt', headless=disable_rendering)
        self.pr.start()

        """ Define action and observation space """
        self.action_space_size = 18
        if experiment_num == 1:
            self.observation_space_size = 59
        elif experiment_num == 2:
            self.observation_space_size = 79
        elif experiment_num == 3:
            self.observation_space_size = 143

        self._load_all_objects()
        self._define_joint_bounds()

        self.previous_distance = None
        self.previous_joint_positions = None

        self.max_tilt_termination = np.deg2rad(90)
        self.velocity_normalizer = RunningMinMaxNormalizer()

        self.max_simulation_count = 50
        self.max_simulation_count_reset = 500
        self.current_global_step_count = 0
        self.current_path = []
        self.current_path_waypoint_index = 0


    def _load_all_objects(self):
        """
        Method responsible for loading all relevant objects from the scene
        :return:
        """
        joint_names = [f'hexa_joint{i}_{j}' for i in range(1, 4) for j in range(6)]
        self.joints = [Joint(name) for name in joint_names]

        all_robot_legs_names = [f'hexa_link{i}Respondable_{j}' for i in range(1, 4) for j in range(6)]
        self.all_robot_legs = [Object.get_object(name) for name in all_robot_legs_names]

        """ Load all sensors """
        self.terrain_reference = Object.get_object("Terrain_reference")
        self.vision_sensor = VisionSensor('Photo_sensor')
        self.terrain_sensor = MyProximitySensor('Terrain_sensor')
        self.vision_terrain_sensor = VisionSensor('Vision_terrain_sensor')
        self.distance_sensor = MyProximitySensor('Proximity_sensor')

        """ Get the appropriate Z coord used for waypoint generation """
        self.sensor_z_coord = self.terrain_sensor.get_position()[2]

        self.robot = Object.get_object('hexapod')
        self.robot_body = Object.get_object('hexa_body')
        self.goal = Object.get_object('Target')
        self.terrain = Object.get_object("Terrain_shape")
        self.wall = Object.get_object("Walls")
        self.base_dimensions = self.wall.get_bounding_box()

        self.robot_feet = []
        self.robot_legs = []
        for i in range(6):
            self.robot_feet.append(Object.get_object(f"hexa_footTip{i}"))
            self.robot_legs.append(Object.get_object(f"hexa_link3_{i}"))

    def _define_joint_bounds(self):
        """
        Method responsible for defining and then for computing each of the joint possible movement range
        :return:
        """
        self.joint_min_max = [(math.radians(-90), math.radians(90)), (math.radians(-90), math.radians(120)),
                              (math.radians(-80), math.radians(175))]
        self.action_bounds_min = np.array([self.joint_min_max[i // 6][0] for i in range(self.action_space_size)],
                                          dtype=np.float32)
        self.action_bounds_max = np.array([self.joint_min_max[i // 6][1] for i in range(self.action_space_size)],
                                          dtype=np.float32)
        self.action_range = [high - low for low, high in zip(self.action_bounds_min, self.action_bounds_max)]

        self.original_joint_positions = [pos.get_position(self.robot_body) for pos in self.joints]

    def set_joint_positions(self, action):
        """
        Method responsible for setting joint target position
        :param action: current action which position is to be set
        :return:
        """
        for joint, value in zip(self.joints, action):
            joint.set_joint_target_position(value)

    def step_and_update(self):
        """
        Method responsible for performing one simulation step
        :return:
        """
        self.pr.step()

    def get_step_results(self, warmup = False):
        """
        Method used for getting all relevant information for performing action in some state of the environment
        :param warmup: boolean flag
        :return: obs, reward, done, goal_reached, reward_components
        """
        obs = self._get_observation()
        done, goal_reached = self._check_termination()
        self._check_waypoint_distance()
        reward, reward_components = self._calculate_reward(warmup)

        return obs, reward, done, goal_reached, reward_components


    def all_joints_reached_targets(self, position_tolerance=0.05):
        """
        Method deciding if all the joints reached their desired target position or if they are not moving anymore
        :param position_tolerance: constant of not moving tolerance
        :return: boolean flag representing if the joint are synced or not
        """
        current_joint_positions = [j.get_joint_position() for j in self.joints]

        for index, j in enumerate(self.joints):
            target = j.get_joint_target_position()

            pos_not_synced = abs(current_joint_positions[index] - target) > position_tolerance

            if self.previous_joint_positions is None:
                if pos_not_synced:
                    self.previous_joint_positions = current_joint_positions
                    return False
            else:
                if pos_not_synced and abs(current_joint_positions[index] - self.previous_joint_positions[index]) > 0.1 :
                    self.previous_joint_positions = current_joint_positions
                    return False

        return True

    @staticmethod
    def _generate_new_position(min_x, max_x, min_y, max_y, offset):
        """
        Method generating new object's position based on dimensions
        :param min_x: x dimension lower bound
        :param max_x: x dimension upper bound
        :param min_y: y dimension lower bound
        :param max_y: y dimension upper bound
        :param offset: position offset
        :return: X and Y coordinate of a randomly generated object's position
        """
        return random.uniform(min_x+offset, max_x-offset), random.uniform(min_y+offset, max_y-offset)

    @staticmethod
    def _change_object_angular_damping(affected_object, new_value):
        """
        Method which calls CoppeliaSim API for angular damping change
        :param affected_object: object which is to be changed
        :param new_value: value of this change
        :return:
        """
        sim.simSetEngineFloatParameter(
            sim.sim_bullet_body_angulardamping,
            affected_object.get_handle(),
            new_value
        )

    @staticmethod
    def _change_object_linear_damping(affected_object, new_value):
        """
        Method which calls CoppeliaSim API for linear damping change
        :param affected_object: object which is to be changed
        :param new_value: value of this change
        :return:
        """
        sim.simSetEngineFloatParameter(
            sim.sim_bullet_body_lineardamping,
            affected_object.get_handle(),
            new_value
        )


    def _randomly_place_robot_target(self):
        """
        Method responsible for random robot and target position placement, also ensuring distance between them
        :return: Robot's and target's positions
        """
        robot_position = self.robot.get_position()
        stable_positions = True

        def _move_object_to_new_position(affected_object, z_position, collision_object):
            while True:
                new_x, new_y = self._generate_new_position(self.base_dimensions[2], self.base_dimensions[3], self.base_dimensions[4], self.base_dimensions[5], 3.0)
                affected_object.set_position([new_x, new_y, z_position])
                self.pr.step()
                if affected_object.check_distance(collision_object) > 1.5:
                    break

        def _ensure_robot_goal_distance():
            while self.robot.check_distance(self.goal) <= 1.5:
                _move_object_to_new_position(self.robot, robot_position[2], self.goal)

        def _calculate_velocities():
            lin_t, ang_t = self.goal.get_velocity()
            lin_r, ang_r = self.robot_body.get_velocity()

            absolute_target_velocity = sum(abs(v) for v in lin_t) + sum(abs(v) for v in ang_t)
            absolute_robot_body_velocity = sum(abs(v) for v in lin_r) + sum(abs(v) for v in ang_r)

            return absolute_target_velocity, absolute_robot_body_velocity

        """ Keep the original Z-coordinate, which is used to set new coordinates """
        _move_object_to_new_position(self.robot, robot_position[2], self.wall)
        _move_object_to_new_position(self.goal, robot_position[2], self.wall)

        """ Ensure that robot and goal are not position right next to each other """
        _ensure_robot_goal_distance()

        absolute_target_velocity, absolute_robot_body_velocity = _calculate_velocities()

        current_simulation_count = 0
        while absolute_target_velocity > 0.001 or absolute_robot_body_velocity > 0.001:
            self.pr.step()

            current_simulation_count += 1
            if current_simulation_count >= self.max_simulation_count_reset:
                stable_positions = False
                break

            absolute_target_velocity, absolute_robot_body_velocity = _calculate_velocities()

        return stable_positions

    def _check_joint_displacement(self):
        """
        Method responsible for catching reposition physics problems before simulation starts
        :return: Boolean flag indicating problem
        """
        for joint, original_joint_pos in zip(self.joints, self.original_joint_positions):
            if np.any(np.abs(joint.get_position(self.robot_body) - original_joint_pos) > 0.2):
                return True
        return False

    def check_robot_collision(self):
        """
       Method responsible for catching physics glitches with the terrain reference model
       :return: Boolean flag indicating problem
       """
        for leg in self.all_robot_legs:
            if leg.check_collision(self.terrain_reference):
                return True
        return False

    def reset(self):
        """
        Method responsible for resetting the simulation
        :return: Current state of the environment
        """
        self.previous_distance = None
        self.previous_joint_positions = None

        self.pr.stop()
        self.pr.start()

        current_simulation_count = 0
        self._change_object_angular_damping(self.robot, 1.0)
        self._change_object_angular_damping(self.goal, 1.0)
        self._change_object_linear_damping(self.robot, 5.0)

        while True:
            have_correct_positions = self._randomly_place_robot_target()
            simulation_problem = self._check_joint_displacement() or self._check_robot_turned_over()
            current_simulation_count += 1
            if have_correct_positions and not simulation_problem:
                break

            self.pr.stop()
            self.pr.start()

        self._change_object_angular_damping(self.robot, 0.0)
        self._change_object_angular_damping(self.goal, 0.0)
        self._change_object_linear_damping(self.robot, 0.0)

        sim.simResetDynamicObject(self.robot.get_handle())
        sim.simResetDynamicObject(self.goal.get_handle())

        """ Waypoint generation """
        self.current_path = self.generate_straight_waypoints(self.robot.get_position(), self.goal.get_position(), 2.0)
        self.current_path_waypoint_index = 0

        observation = self._get_observation()

        if self.experiment_num == 1:
            self.previous_distance = np.linalg.norm(self.goal.get_position() - self.robot.get_position())
        elif self.experiment_num == 2 or self.experiment_num == 3:
            self.previous_distance = np.linalg.norm(self._get_current_waypoint() - self.robot.get_position())

        return observation


    def _calculate_terrain_z_coord(self):
        """
        Method used to calculate z coordinate of point of contact with the terrain
        :return: Z coordinate which corresponds to point of contact on the terrain
        """
        _, distance, point, _ = self.terrain_sensor.read()

        if distance == -1:
            self.results_queue.put(f"Terrain distance error")

        return self.terrain_sensor.get_position()[2] - distance
    def generate_straight_waypoints(self, start, goal, step_size):
        """
        Method responsible for generating dynamically chanding number of waypoints along the path between robot and goal initial position
        :param start: current robot position
        :param goal: current goal position
        :param step_size: constant influencing the number of waypoint to be generated
        :return: List of waypoints
        """
        path = []
        direction = goal - start
        distance = np.linalg.norm(direction)

        direction_norm = direction / distance
        possible_steps = int(distance // step_size)
        for i in range(1, possible_steps + 1):
            point = start + direction_norm * (i * step_size)

            self.terrain_sensor.set_position([point[0], point[1], self.sensor_z_coord])
            self.pr.step()
            point[2] = self._calculate_terrain_z_coord() + 0.3
            path.append(point)
        path.append(goal)

        return path

    def _get_current_waypoint(self):
        """
        Get current waypoint from the list
        :return: Current waypoint [x,y,z] position
        """
        return self.current_path[self.current_path_waypoint_index]

    def _check_waypoint_distance(self):
        """
        Method responsible for waypoint advancement based on the difference of positions between robot and it
        :return:
        """
        current_robot_pos = self.robot.get_position()

        if (self.current_path_waypoint_index < len(self.current_path) and
                np.linalg.norm(self.current_path[self.current_path_waypoint_index] - current_robot_pos) < 0.5):
            self.current_path_waypoint_index += 1

    def _calculate_robot_velocity_simple(self, goal_pos, robot_body_pos):
        """
        Legacy method used in the first experiment to calculate velocity instead of distance based principle used later
        :param goal_pos: current robot position
        :param robot_body_pos: current robot body position
        :return: Float value representing the robot's velocity in the direction of the goal
        """
        direction = goal_pos - robot_body_pos
        direction_to_goal_normalized = direction / np.maximum(np.linalg.norm(direction), 1e-8)
        linear, _ = self.robot_body.get_velocity()
        velocity_toward_goal = float(np.dot(linear, direction_to_goal_normalized))

        return velocity_toward_goal

    def _get_observation(self):
        """
        Method reponsible for getting observation space entries
        :return:
        """
        num_joints = len(self.joints)
        joint_velocities = np.zeros(num_joints)
        positions = np.zeros(num_joints)
        """ Get joint velocities and their current positions """
        for i, joint in enumerate(self.joints):
            joint_velocities[i] = joint.get_joint_velocity()
            positions[i] = joint.get_joint_position()

        robot_pos = self.robot_body.get_position()
        terrain_normal = self._get_normal_from_terrain()
        if terrain_normal is None:
            terrain_normal = [0,0,1]

        roll, pitch, yaw = self.robot_body.get_orientation()
        R_r = self.robot_body.get_matrix()[:3, :3]

        """ Calculate the current goal direciton vector """
        if self.experiment_num == 1:
            direction = self.goal.get_position() - robot_pos
            direction_to_goal_normalized = direction / np.maximum(np.linalg.norm(direction), 1e-8)

        elif self.experiment_num == 2 or self.experiment_num == 3:
            direction = self._get_current_waypoint() - robot_pos
            direction_to_goal_normalized = direction / np.maximum(np.linalg.norm(direction), 1e-8)

        """ Transform these observations into robot's body space"""
        linear_world, angular_world = self.robot_body.get_velocity()
        linear_body = R_r.T @ linear_world
        angular_body = R_r.T @ angular_world
        target_dir_body = R_r.T @ direction_to_goal_normalized

        """ Get foot velocities and their distances from terrain """
        foot_velocities = []
        foot_contacts = np.zeros(6)
        for i, foot in enumerate(self.robot_feet):
            foot_velocities.extend(foot.get_velocity()[0])
            foot_contacts[i] = max(foot.check_distance(self.terrain) - 0.0125, 0)

        """ Get current robot's orientation """
        hexa_orient = np.array([
            np.sin(roll), np.cos(roll),
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw)
        ])

        """ Get its current height from terrain """
        _, body_height, _, _ = self.distance_sensor.read()
        depth_map = None
        if self.experiment_num == 3:
            """ Capture depth map of space below his body """
            self.vision_terrain_sensor.handle_explicitly()
            depth_map = self.vision_terrain_sensor.capture_depth()
            depth_map = depth_map.flatten()

        if self.experiment_num == 1 :
            target_dir_velocity = self._calculate_robot_velocity_simple(self.goal.get_position(), self.robot_body.get_position())
            observation = np.concatenate(
                (direction_to_goal_normalized, [target_dir_velocity], positions, joint_velocities,
                 hexa_orient, linear_world, angular_world, [body_height], foot_contacts))
        elif self.experiment_num == 2:
            observation = np.concatenate((target_dir_body, terrain_normal, positions, joint_velocities, hexa_orient, linear_body, angular_body, [body_height], foot_contacts, foot_velocities))
        elif self.experiment_num == 3:
            observation = np.concatenate((target_dir_body, terrain_normal, positions, joint_velocities, hexa_orient, linear_body, angular_body, [body_height], foot_contacts, foot_velocities, depth_map))


        return observation.astype(np.float32)


    def _calculate_robot_distance_progress(self, goal_pos, robot_body_pos):
        """
        Method which purpose it calculating difference between new and previous distance from the current goal
        :param goal_pos: position of the current goal
        :param robot_body_pos: current robot position
        :return: Value representing this position difference
        """
        current_distance = np.linalg.norm(goal_pos - robot_body_pos)
        progress = self.previous_distance - current_distance
        self.previous_distance = current_distance
        return progress

    def _get_normal_from_terrain(self):
        """
        Method responsible for calculating normal from the terrain surface
        :return: Normal vector or None
        """
        _, _, _, normal = self.distance_sensor.read()

        if normal is not None:
            R_s = self.distance_sensor.get_matrix()[:3, :3]
            """ Distance sensor normal is in it's own sensor space -> need to cast to world space """
            n = R_s @ normal

            n /= np.maximum(np.linalg.norm(n), 1e-8)
            return n

        return None

    def _get_robot_tilt(self):
        """
        Method responsible for calculating current robot tilt based on the difference between terrain normal and its own Z coordinate
        :return: Dot product between these unit vectors
        """
        n = self._get_normal_from_terrain()

        if n is not None:
            R_r = self.robot_body.get_matrix()[:3, :3]
            z = R_r[:, 2]

            z /= np.maximum(np.linalg.norm(z), 1e-8)
            cos_theta = np.dot(z, n)
            return max(0, cos_theta)
        return 0


    def _calculate_reward(self, warmup):
        """
        Method responsible for calculating reward based on the current state of hexapod
        :param warmup: boolean flag indicating warmup phase
        :return:
        """
        distance_progress = 0
        if self.experiment_num == 1:
            distance_progress = self._calculate_robot_distance_progress(self.goal.get_position(), self.robot_body.get_position())
        elif self.experiment_num == 2 or self.experiment_num == 3:
            distance_progress = self._calculate_robot_distance_progress(self._get_current_waypoint(), self.robot_body.get_position())

        if warmup:
            self.velocity_normalizer.update(distance_progress)
        velocity_reward = self.velocity_normalizer.normalize(distance_progress)

        v_target = 0.2
        """ Velocity reward shaping """
        if v_target >= velocity_reward >= 0:
            velocity_reward = velocity_reward / v_target
        elif velocity_reward > v_target:
            velocity_reward = max(1.0 - 2*(velocity_reward - v_target), -1)
        else:
            velocity_reward = max(velocity_reward / v_target, -1)

        roll, pitch, _ = self.robot_body.get_orientation()
        leg_tip_reward = 0

        proximities = []
        for i in range(6):
            d = max(self.robot_feet[i].check_distance(self.terrain) - 0.0125, 0)
            """ We allow little measuring error"""
            if d < 0.005:
                d = 0

            proximities.append(min(d, 1.0))
            d_max = 1.0
            d = min(d, d_max)
            reward = -(1 / 6) * (d / d_max)

            leg_tip_reward += reward

        tilt = self._get_robot_tilt()
        tilt_reward = tilt - 1.0

        R_r = self.robot_body.get_matrix()[:3, :3]
        normal = R_r[:, 2]
        normal /= np.maximum(np.linalg.norm(normal), 1e-8)

        leg_tilt_reward = -1
        if normal is not None:
            leg_tilt_reward = 0
            for index, (leg, tip) in enumerate(zip(self.robot_legs, self.robot_feet)):
                current_leg_pos = leg.get_position()
                current_leg_tip_pos = tip.get_position()
                direction_of_the_leg = current_leg_pos - current_leg_tip_pos
                direction_of_the_leg /= np.maximum(np.linalg.norm(direction_of_the_leg), 1e-8)

                result = np.dot(direction_of_the_leg, normal)
                result = np.clip(result, 0, 1)

                result = result - 1

                effective_result = result + proximities[index] * (-1 - result)
                leg_tilt_reward += (1 / 6) * effective_result

        _, z, _, _ = self.distance_sensor.read()

        if z != -1.0:
            target_height = 0.25
            body_height_reward = -min(abs(z - target_height), 1.0)
        else:
            body_height_reward = -1

        fell_over_reward = 0
        if self._check_robot_turned_over():
            fell_over_reward = -5

        stability_reward = (
                0.4 * tilt_reward +
                0.6 * body_height_reward
        )
        legs_reward = (
                0.05 * leg_tip_reward +
                0.95 * leg_tilt_reward
        )
        motion_reward = (
                1.0 * velocity_reward
        )

        reward = 1.0*stability_reward + 1.0*legs_reward + 2*motion_reward + fell_over_reward

        return reward, (1.0*stability_reward, 1.0*legs_reward, 2*motion_reward, fell_over_reward)

    def _check_robot_turned_over(self):
        """
        Method responsible for checking if the robot orientation is over the maximum limit
        :return: Boolean flag indicating robot being turned over
        """
        roll, pitch, _ = self.robot_body.get_orientation()
        return abs(roll) > self.max_tilt_termination or abs(pitch) > self.max_tilt_termination

    def _check_termination(self):
        """
        Method responsible for episode termination - in case of turning over and in case of reaching the target
        :return: Two boolean values, first one represent incorrect environment state, the second one represents robot reaching target position
        """
        if self._check_robot_turned_over():
            return True, False

        if self.robot.check_distance(self.goal) < 1.0:
            try:
                if self.disable_rendering:
                    self.results_queue.put_nowait("Robot reached target")
            except Full:
                pass
            return False, True

        return False, False

    def close(self):
        """
        Handling PyRep simulation termination
        :return:
        """
        self.pr.stop()
        self.pr.shutdown()
