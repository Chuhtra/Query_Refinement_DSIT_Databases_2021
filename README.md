# Query_Refinement_DSIT_Databases_2021

# Introduction

This is a Python CLI application implemented in the context of the DSIT Databases course project at National Kapodistrian University of Athens. It is a query solver for Constraint Programming queries for time series data, with automatic query refinement based on the [Dynamic Query Refinement for Interactive Data Exploration, Kalinin et. al](https://openproceedings.org/2020/conf/edbt/paper_25.pdf) paper. The implementation is based on a custom Binary Search Tree for solving CP problems with the backtracking algorithm. Data are stored and retrieved using PostgreSQL.

# Datasets used

For the experiments and test runs two datasets were used, both downloaded from Kaggle:

- Electromyography Signal Data acquired with Myo Gesture Control Armband ([link](https://www.kaggle.com/cozer1987/emg-dataset-taken-with-myo-armband)) / EMG
- Gas sensor array temperature modulation ([link](https://www.kaggle.com/javi2270784/gas-sensor-array-temperature-modulation)) / GAS

To use these datasets, please download the zip files in an **empty** directory, and then follow the guidelines for setting up the testing environment.

# Environment Setup

Let it be noted that the following steps were tested only on a Manjaro Linux machine. Download the source files in a directory.

## Python

To be able to run the program a python environment with the packages documented in _requirements.txt_ file needs to be set up.
Supposing that [Python](https://www.python.org/downloads/release/python-397/) is already installed (v.3.9 was used), a clean way to do this is by using the [venv](https://docs.python.org/3/library/venv.html) package as follows:

Open a terminal in your Desktop.

- Run `pip install virtualenv` if needed.
- Run `mkdir db_dsit_env_directory` to create a directory to store the virtual environment in
- Run `cd db_dsit_env_directory` to change working directory
- Run `virtualenv -p python3.9 db_dsit_env` to create a Pyhon 3.9 environment.
- Run `source db_dsit_env/bin/activate` to activate the environment.
- Run `cd path_to_src` to change working directory to the `src` directory you downloaded from GitHub.
- Run `pip install -r ../requirements.txt` to install needed packages


## Postgres

Supposing that the Python environment is ready and PostgreSQL is already installed (v.13.3 was used), the only thing needed to set up the datasets is to run the `db_preparation.py` script with this command:

`python3 db_preparation.py --filepath path_to_dir_with_zip_files`

Notes:

- The `db_preparation.py` script creates two databases (one for each dataset) in the local Postgres installation and loads the data in tables (one for each database) to be ready for querying.
- After running this command, if you want to run it again make sure that the directory with the zip files contains nothing but the zip files, and also run the `post_cleanup.py` script once.


# How to use

## Run a query

1. Please make sure there is an `obj` directory on the same path level with the `src` source directory. It is needed to store program results.
2. Use the default query or add a query from the examples in the report to the `query.txt` file, and then run `python3 client.py`
3. Head to the `obj` directory to see the drawn plot.


## Formulating queries

If not sure about query structure or come upon a breakage case that isn't noted below, please use the examples that can be found in the report Appendix.

Query MUST be of the following format:
```
SELECT time_id, offset IN_DOMAIN [d, d], [d, d]
FROM table.column
WHERE constraint([arguments]) in [d, d] [MAX|MIN]
            [ and ... ]
[LIMIT [REFINED] 50]
```

Notes:

1. d can be changed with None, to provide unbounded domain sides, or INTEGERS.
2. REFINED can be omitted to get limited unrefined results only.
3. Queries shouldn't end with ';' like normal SQL, since ';' lead to untested behavior.
4. LIMIT line can be omitted entirely to get all results that fit the unrefined query.
5. Except for WHERE constraints, changing line structure leads to untested behavior.
6. constraint_name can be one of the avg_amp, max_amp_excess_left or max_amp_excess_right.
7. The last two need 1 integer argument, that should be given in the parenthesis.
8. target can be either MAX or MIN.
9. Query must contain one table and one column. No more.


## How to run experiments

To do the experiments simply run the program for the selected queries described in the report and collect the data. 

To run the manual tightening case, a specific part in `execute_query()` method of the `server.py` file needs to be un-commented.

The `scalability_experiment_*.py` files are used to run the repeated refined query runs and plots respectively, for the scalability experiments. These _do not_ run out of the box, like the implemented system.

# Environment cleanup

To undo all the steps documented above simply take these steps:

- Run `python3 post_cleanup.py` script to undo the database creations of the `db_preparation.py` script.
- Run `deactivate` to deactivate the virtual environment.
- Delete the directory where the python virtual environment is stored. (`db_dsit_env_directory` if previous guide was followed)
- Delete the directory with the downloaded and extracted dataset zip-files and the one with the source files.
