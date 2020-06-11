from typing import Mapping

from blockflow.hooks import Hooks
from blockflow.experiment import Experiment
from blockflow.contract.ethereum_address import EthereumAddress

from .model import LogisticRegressionEvaluationModel
from .dataset import LogisticRegressionDataset

class LRHooks(Hooks):
    def __init__(self, ground_truth_dataset: LogisticRegressionDataset) -> None:
        self._ground_truth_dataset = ground_truth_dataset

    def post_train(self, dp_round: int, experiment: Experiment) -> None:
        model = experiment._model_factory.training_model
        assert isinstance(model, LogisticRegressionEvaluationModel)
        score = model.f1_score(self._ground_truth_dataset)
        assert experiment._statistics_recorder is not None
        experiment._statistics_recorder.record_scalar(dp_round, "my_model_ground_truth_dataset", score)

    def post_score(self, dp_round: int, experiment: Experiment, scores: Mapping[EthereumAddress, float]) -> None:
        model = experiment._model_factory.training_model
        assert isinstance(model, LogisticRegressionEvaluationModel)
        score = model.f1_score(self._ground_truth_dataset)
        assert experiment._statistics_recorder is not None
        experiment._statistics_recorder.record_scalar(dp_round, "shared_model_ground_truth_dataset", score)
