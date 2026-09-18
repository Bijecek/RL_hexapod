from TD3.td3 import TD3
from TD3.memory import Memory
from TD3.td3_logic import TD3Logic
from TD3.teacher import Teacher
from TD3.worker import Worker
from hexapod_env import HexapodEnv
import multiprocessing as mp
import os.path
import threading
from queue import Empty
from TD3.ou_noise import Noise

class ExperimentRunner:
    def __init__(self, directory_name, n_workers = 8, experiment_num=0):
        """
        Initialization of the experiment runner
        :param directory_name: name of the directory in which the results will be saved
        :param n_workers: desired number of worker processes
        :param experiment_num: chosen experiment number
        """
        self.directory_name = directory_name
        self.n_workers = n_workers
        self.experiment_num = experiment_num

        self.exp_queue = None
        self.weights_queues = []
        self.control_queues = []
        self.results_queue = None
        self.workers = []
        self.printer_thread = None

    def run_training(self):
        """
        Method containing all the program's lifecycle
        :return:
        """
        self._prepare_directory()
        self._setup_multiprocessing()
        self._create_queues()
        self._start_workers(self.experiment_num)
        self._start_printer()
        self._start_teacher()

        self._wait_and_close()

    def _prepare_directory(self):
        while True:
            if os.path.exists(self.directory_name):
                print(f"Directory {self.directory_name} already exists. Do you want to overwrite it? (y/n)")
                if input().lower() == "y":
                    os.makedirs(self.directory_name, exist_ok=True)
                    break
                else:
                    self.directory_name = input(f"Plase provide new directory name: ")
            else:
                break

    def _setup_multiprocessing(self):
        mp.set_start_method("spawn", force=True)
        print(f"Starting with {self.n_workers} workers")

    def _create_queues(self):
        """
        Method responsible for queue creation
        :return:
        """
        self.exp_queue = mp.Queue(maxsize=1_000)
        self.weights_queues = [mp.Queue(maxsize=1) for _ in range(self.n_workers)]
        self.control_queues = [mp.Queue(maxsize=10) for _ in range(self.n_workers)]
        self.results_queue = mp.Queue(maxsize=100)

    @staticmethod
    def _run_worker(directory_name, exp_queue, weights_queue, control_queue, results_queue, worker_id, experiment_num):
        """
        Method responsible for initializing and executing Worker process
        :param directory_name: name of the directory in which the results will be saved
        :param exp_queue: experience queue in which each worker inserts feedback from env
        :param weights_queue: a queue used for receiving new actor weights
        :param control_queue: synchronization queue used between worker and teacher
        :param results_queue: results queue in which each worker inserts current episode performance
        :param worker_id: current worker identification number
        :param experiment_num: chosen experiment number
        :return:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        env = HexapodEnv(results_queue, disable_rendering=True, experiment_num = experiment_num)
        td3 = TD3(env.observation_space_size, env.action_space_size, env.action_bounds_min, env.action_bounds_max, experiment_num)

        if experiment_num == 1 or experiment_num == 2:
            noise = Noise(
                std_start=0.15,
                std_end=0.05,
                max_steps=500 * 150,
                theta=0.15, dt=1.0,
                noise_shape=(env.action_space_size,),
                action_range=env.action_range)
        elif experiment_num == 3:
            noise = Noise(
                std_start=0.15,
                std_end=0.05,
                max_steps=1000 * 150,
                theta=0.15, dt=1.0,
                noise_shape=(env.action_space_size,),
                action_range=env.action_range)

        worker = Worker(directory_name, exp_queue, weights_queue, control_queue, results_queue, worker_id, env, td3,
                        noise)
        worker.run_episodes(max_episodes=40_000, max_steps=150, warm_up_memory=500)

    @staticmethod
    def _run_teacher(exp_queue, weight_queues, control_queues, current_num_of_workers, directory_name, experiment_num):
        """
        Method responsible for initializing and executing Teacher process
        :param exp_queue: experience queue from which the teacher gathers feedback from each worker's interactions
        :param weight_queues: a queues used for sending each worker new actor weights
        :param control_queues: synchronization queues used between worker and teacher
        :param current_num_of_workers: current number of workers
        :param directory_name: name of the directory in which the results will be saved
        :return:
        """
        env = HexapodEnv(None, disable_rendering=True, experiment_num=experiment_num)
        td3 = TD3Logic(env.observation_space_size, env.action_space_size, env.action_bounds_min, env.action_bounds_max,
                       0.0001, 0.001, gamma=0.99, tau=0.005, experiment_num=experiment_num)

        memory = Memory(1_000_000, env.observation_space_size, env.action_space_size, batch_size=256)

        teacher = Teacher(exp_queue, weight_queues, control_queues, td3, memory, current_num_of_workers,
                          directory_name, update_rate=100)
        teacher.teacher_loop()

    @staticmethod
    def _print_results(results_queue):
        """
        Method used for printing each of the worker's results
        :param results_queue: results queue in which each worker inserts current episode performance
        :return:
        """
        while True:
            try:
                msg = results_queue.get(timeout=0.5)
                if msg == "STOP printer":
                    print("STOPPING printer")
                    break
                else:
                    print(msg)
            except Empty:
                pass
            except Exception as e:
                break

    def _start_workers(self, experiment_num):
        for wid in range(self.n_workers):
            p = mp.Process(
                target=self._run_worker,
                args=(self.directory_name, self.exp_queue, self.weights_queues[wid], self.control_queues[wid], self.results_queue, wid, experiment_num)
            )
            p.start()
            self.workers.append(p)

    def _start_printer(self):
        printer_thread = threading.Thread(
            target=self._print_results,
            args=(self.results_queue,),
        )
        printer_thread.start()

    def _start_teacher(self):
        self._run_teacher(self.exp_queue, self.weights_queues, self.control_queues, self.n_workers, self.directory_name, self.experiment_num)

    def _wait_and_close(self):
        """
        Method responsible for cleanup logic
        :return:
        """
        for id, p in enumerate(self.workers):
            print(f"Waiting for {id}")
            p.join()

        print("Workers finished")
        self.exp_queue.close()
        self.exp_queue.join_thread()

        print("EXP QUEUE CLOSED")
        for q in self.weights_queues:
            q.cancel_join_thread()
            q.close()

        print("WEIGHT CLOSED")

        for q in self.control_queues:
            q.close()
            q.join_thread()

        print("CONTROL CLOSED")

        self.results_queue.close()

        self.results_queue.join_thread()

        print("RESULT CLOSED")