# BlockFLow

To use BlockFLow for your own federated, differentially private machine learning project, you'll first need to implement the following classes:

* In [model.py](model.py): `EvaluationModel`, `TrainingModel`, and `ModelFactory`
* In [contract/web3_factory.py](contract/web3_factory.py): `Web3Factory`

You will also need to provide the parameters specified in [client_config.py](client_config.py)