# BlockFLow Supplementary materials

## Contents
Our supplementary materials submission contains the following:
1. [Paper with Appendix](paper_with_appendix.pdf)
1. Data for all experiments in the [results](results) folder
1. Code to reproduce figures that appear in the paper in the [experiments_and_figures](experiments_and_figures) folder
1. Source code in the [src] folder
1. Intructions to reproduce results, below:


## Result Reproduction Instructions

1. Install [Python 3.8](https://www.python.org/downloads/release/python-380/) on Linux. We have not tested BlockFLow on other platforms.
1. Run `pip install -r src/requirements.txt`
1. Run all cells in [src/examples/logistic_regression/adult/generate_datasets.ipynb](src/examples/logistic_regression/adult/generate_datasets.ipynb), specifying paths as appropriate
1. Run all cells in [src/examples/logistic_regression/kdd99/generate_datasets.ipynb](src/examples/logistic_regression/kdd99/generate_datasets.ipynb), specifying paths as appropriate
1. Install [Docker](https://www.docker.com/get-started)
1. Results in experiments 1-4 can be reprocduced through off-chain simulation of BlockFLow. To do so,
    1. Run all cells in [experiments_and_figures/exp1_thru_4.ipynb](experiments_and_figures/exp1_thru_4.ipynb), specifying paths as appropriate
    1. For experiments 1, 2, or 4:
        1. Run all cells in `experiments_and_figures/exp{NUMBER}.ipynb`, where NUMBER is 1, 2, or 4. Specify paths as appropriate
    1. For experiment 3:
        1. Run all cells in [experiments_and_figures/score_exp3.ipynb](experiments_and_figures/score_exp3.ipynb), specifying paths as appropriate
        1. Run all cells in [experiments_and_figures/exp3.ipynb](experiments_and_figures/exp3.ipynb), specifying paths as appropriate
1. Results in experiment 5 can only be reporduced through an on-chain simulation of BlockFLow. To do so,
    1. Install [Docker Compose](https://docs.docker.com/compose/install/)
    1. Copy [src/examples/example.config.py](src/examples/example.config.py) to `src/examples/config.py`. Set `RESULTS_FOLDER_PATH` to some empty folder
    1. Run all cells in [src/infra/geth/generate_gensis.ipynb](src/infra/geth/generate_gensis.ipynb)
    1. Copy the JSON-stringification of `genesis` to [src/infra/geth/genesis.json](src/infra/geth/genesis.json)
    1. Copy the value of the `private_keys` variable to the value of `PRIVATE_KEYS` in `src/examples/config.py`
    1. From `src/infra/local`, run `docker-compose up --build`
    1. From `src`, run `python3 -m examples.logistic_regression.run_multi_clients --exp_name adult --dataset_folder /path/to/data/root/adult/split_sym_eq_1_validation_fraction_0.2 --ground_truth_dataset /path/to/data/root/adult/test.dat`
    1. From `src`, run `python3 -m examples.logistic_regression.run_multi_clients --exp_name adult --dataset_folder /path/to/data/root/adult/split_sym_eq_3_validation_fraction_0.2 --ground_truth_dataset /path/to/data/root/adult/test.dat`
    1. From `src`, run `python3 -m examples.logistic_regression.run_multi_clients --exp_name adult --dataset_folder /path/to/data/root/adult/split_sym_eq_5_validation_fraction_0.2 --ground_truth_dataset /path/to/data/root/adult/test.dat`
    1. Inside `RESULTS_FOLDER_PATH/tensorboard`, run `tensorboard --logdir .`
    1. Tensorboard visualizes the gas consumption by federated learning round and provides links to export such data into CSV or JSON format. Copy data into statistical software to perform linear regressions.
