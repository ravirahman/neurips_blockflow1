# Examples

Multiple examples of how to use BlockFLow in a machine learning project are provided:

* Linear Regression:
  * [Simple](linear_regression/simple/): A very simple 2 feature linear regression challenge. Useful for validating the setup of the system (e.g. IPFS and Ethereum)
* Logistic Regression:
  * [Adult](logistic_regression/adult/): A logistic regression classifier for the [UCI Adult](http://archive.ics.uci.edu/ml/datasets/Adult) dataset.
  * [Kdd99](logistic_regression/kdd99/): A logistic regression classifier for the [KDD99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) challenge.

## How to run
1. First, copy [example.config.py](example.config.py) to `config.py`, and populate the values as appropriate.
   1. Note: if you need an Ethereum node and IPFS host, see the [infrastructure templates](../infra/)
2. Go to the [root directory of the repo](../) and run `pip install -r requirements.txt`
3. For the linear regression example:
   1. Run, from the [root directory of the repo](../), `python -m examples.linear_regression.simple.run_single_client`. This example will run one client
   2. You can also simulate a multiple-client workflow with `python -m examples.linear_regression.simple.run_multi_clients`. This example uses python's `subprocessing` to spawn multiple clients on the same machine.
4. For either of the logistic regression examples:
   1. First, you will need to configure the dataset. See the jupyter notebooks for [adult](logistic_regression/_datasets/adult/generate_datasets.ipynb) and [kdd99](logistic_regression/_datasets/kdd99/generate_datasets.ipynb). These notebooks allow you to define how to split the datasets for multiple clients. They will output the data into a folder hierarchy of `/path/to/dataset_split_name/client_i/dataset_name.dat`, where `dataset_split_name` is either `train`, `validation`, or `score`.
   2. To run for a single client, from the [root directory of the repo](../), run `python -m examples.logistic_regression.run_single_client adult /path/to/dataset_split_name/client0`. You can replace `adult` with `kdd99`
   3. To run for multiple clients, from the [root directory of the repo](../), run `python -m examples.logistic_regression.run_multi_clients adult /path/to/dataset_split_name`. You can replace `adult` with `kdd99`. Like the linear regression example, it uses python's subprocessing to spawn multiple clients
