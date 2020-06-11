from typing import Sequence, cast
from datetime import datetime
import logging

import numpy as np

from blockflow.model import TrainingModel, EvaluationModel, ModelFactory, OthersWork

from .dataset import MyDataset, MyDatasetFactory

class InvalidShapeError(Exception):
    pass

class InvalidDTypeError(Exception):
    pass

class InvalidTypeError(Exception):
    pass

class OtherValidationError(Exception):
    pass

class MyEvaluationModel(EvaluationModel):
    def __init__(self, weights: np.ndarray, validation_dataset: MyDataset) -> None:
        self._weights = weights
        self._logger = logging.getLogger(__name__)
        self._validation_dataset = validation_dataset
        if not isinstance(self._weights, np.ndarray):
            raise InvalidTypeError()
        if not self._weights.shape == (2,):
            raise InvalidShapeError()
        if not self._weights.dtype == np.float:
            raise InvalidDTypeError()

    def save(self, filename: str, mode: str = "xb") -> None:
        with open(filename, mode) as f:
            np.savez(f, weights=self._weights)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def evaluate(self, *datasets: MyDataset) -> Sequence[float]:
        my_datasets = cast(Sequence[MyDataset], datasets)
        x_s = np.stack(tuple(my_dataset.x_s for my_dataset in my_datasets))
        y_s = np.stack(tuple(my_dataset.y_s for my_dataset in my_datasets))
        num_clients = len(datasets)
        n = x_s.shape[1]
        assert x_s.shape == (num_clients, n, 2)
        assert y_s.shape == (num_clients, n, 1)
        assert self._weights.shape == (2,)
        weights_2d = np.expand_dims(self._weights, 1)
        assert weights_2d.shape == (2, 1)
        y_pred = x_s @ weights_2d
        assert y_pred.shape == y_s.shape
        mse = np.square(y_pred - y_s).mean(axis=(1, 2))
        assert mse.shape == (num_clients,)
        quality = 1.0 / np.cosh(mse)
        assert quality.shape == mse.shape
        return cast(Sequence[float], quality.tolist())

    def score(self) -> float:
        return self.evaluate(self._validation_dataset)[0]

class MyTrainingModel(TrainingModel, MyEvaluationModel):
    def __init__(self, weights: np.ndarray, training_dataset: MyDataset, validation_dataset: MyDataset) -> None:
        super().__init__(weights, validation_dataset)
        self._training_dataset = training_dataset

    def _train_step(self, learning_rate: float) -> None:
        num_rows = len(self._training_dataset)
        x_s = self._training_dataset.x_s
        y_s = self._training_dataset.y_s
        assert x_s.shape == (num_rows, 2)
        assert y_s.shape == (num_rows, 1)
        result = np.expand_dims(np.matmul(x_s, self._weights.T), axis=1)
        assert result.shape == (num_rows, 1)
        # error = ((result - y_s)**2).sum()
        derror = 2 * (result - y_s)
        backprop = derror * x_s
        assert backprop.shape == (num_rows, 2)
        backprop_scaled = - learning_rate * backprop
        backprop_averaged = np.mean(backprop_scaled, axis=0, keepdims=True)
        self._weights[...] += np.squeeze(backprop_averaged)

    def update(self, dp_round: int, others_work: Sequence[OthersWork]) -> None:
        # doing a weighted average, where each model is weighted in proportion to its score
        total_score = 0.0
        assert len(others_work) > 0, "cannot call on 0 models"
        weights = np.zeros(2,)
        for dummy_client, model, score in others_work:
            assert isinstance(model, MyEvaluationModel)
            total_score += score
            weights += model.weights * score
        weights /= total_score
        self._weights[...] = weights

    def train(self, dp_round: int, deadline: datetime) -> None:
        step = 0
        while datetime.now() < deadline:
            learning_rate = 0.1
            if step > 3:
                learning_rate = 0.01
            if step > 6:
                learning_rate = 0.001
            step += 1
            self._train_step(learning_rate)

class MyModelFactory(ModelFactory):
    def __init__(self, training_dataset: MyDataset, validation_dataset: MyDataset):
        self._model = MyTrainingModel(np.array((0.234541293874, 0.9174718866524)), training_dataset, validation_dataset)
        self._validation_dataset = MyDatasetFactory().generate(100)

    def load(self, filename: str) -> MyEvaluationModel:
        with np.load(filename) as data:
            return MyEvaluationModel(data['weights'], self._validation_dataset)

    @property
    def training_model(self) -> MyTrainingModel:
        return self._model
