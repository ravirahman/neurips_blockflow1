from typing import TYPE_CHECKING, Mapping

from .contract.ethereum_address import EthereumAddress
from .model import EvaluationModel

if TYPE_CHECKING:
    from .experiment import Experiment

class Hooks:
    def pre_enrollment(self, experiment: 'Experiment') -> None:
        pass

    def post_enrollment(self, experiment: 'Experiment') -> None:
        pass

    def pre_train(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_train(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def pre_data_retrieval(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_data_retrieval(self, dp_round: int, experiment: 'Experiment', data: Mapping[EthereumAddress, EvaluationModel])-> None:
        pass

    def pre_score(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_score(self, dp_round: int, experiment: 'Experiment', scores: Mapping[EthereumAddress, float]) -> None:
        pass

    def pre_score_decrypt(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_score_decrypt(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def pre_post_submit(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_post_submit(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def pre_collect_reward(self, dp_round: int, experiment: 'Experiment') -> None:
        pass

    def post_collect_reward(self, dp_round: int, experiment: 'Experiment') -> None:
        pass
