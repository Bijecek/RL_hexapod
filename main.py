import sys
from TD3.experiment_runner import ExperimentRunner
from TD3.test_runner import TestRunner

def main():
    if len(sys.argv) == 4:
        directory_name = sys.argv[1]
        mode = sys.argv[2]
        experiment_num = int(sys.argv[3])

        if experiment_num == 1 or experiment_num == 2 or experiment_num == 3:

            """ Training """
            if mode == "Train":
                print(directory_name)
                experiment_runner = ExperimentRunner(directory_name, n_workers=8, experiment_num=experiment_num)
                experiment_runner.run_training()
            elif mode == "Test":
                """ Testing """
                test_runner = TestRunner(directory_name, experiment_num = experiment_num)
                test_runner.run_test(max_steps=100)

if __name__ == '__main__':
    main()