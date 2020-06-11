from abc import ABC, abstractmethod
from typing import Sequence, NamedTuple, Optional
from datetime import datetime

from .contract.ethereum_address import EthereumAddress

class EvaluationModel(ABC):
    @abstractmethod
    def score(self) -> float:
        # returns a score between 0 and 1, where 0 is bad and 1 is perfect
        pass

    @abstractmethod
    def save(self, filename: str, mode: str = "xb") -> None:
        # save the model to filename
        pass

class OthersWork(NamedTuple):
    client: EthereumAddress
    model: EvaluationModel
    score: float

class TrainingModel(EvaluationModel, ABC):
    @abstractmethod
    def update(self, dp_round: int, others_work: Sequence[OthersWork]) -> None:
        pass

    @abstractmethod
    def train(self, dp_round: int, deadline: Optional[datetime] = None) -> None:
        # Should train and update the internal model
        # Should apply DP, as appropriate
        pass

class ModelFactory(ABC):
    @abstractmethod
    def load(self, filename: str) -> EvaluationModel:
        # Load is responsible for validating the model in filename
        pass

    @property
    @abstractmethod
    def training_model(self) -> TrainingModel:
        # Should return the training model
        pass
