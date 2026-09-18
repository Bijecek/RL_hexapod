After installing all requirements, the program can be executed in two ways -- Training and Testing phase.

For training phase, empty folder will be created or if agreed to the promp, current folder will be overwriten.

For testing phase, min_max_normalizer_data.pkl and running_normalizer_data.pkl and hexapod_actor.h5 need to be present in the folder.

User can execute training phase for experiments 1, 2 and 3.
Data for testing phase are available only from experiments 1 and 2.
 
Train:
python3 main.py RESULTS_DIRECTORY_NAME Train EXPERIMENT_NUMBER

Test:
python3 main.py DATA_DIRECTORY_NAME Test EXPERIMENT_NUMBER
