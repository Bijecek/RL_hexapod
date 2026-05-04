from TD3.td3 import TD3
from TD3.worker import Worker
from hexapod_env import HexapodEnv


class TestRunner:
    def __init__(self, directory_name, experiment_num):
        self.directory_name = directory_name
        self.experiment_num = experiment_num

    def run_test(self, max_steps):
        env = HexapodEnv(None, disable_rendering=False, experiment_num = self.experiment_num)
        td3 = TD3(env.observation_space_size, env.action_space_size, env.action_bounds_min, env.action_bounds_max, self.experiment_num)

        worker = Worker(self.directory_name, None, None, None, None, None, env, td3,
                        None)
        worker.run_test(max_steps=max_steps)