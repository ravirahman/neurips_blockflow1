from typing import Sequence, Dict, NamedTuple, Optional
from datetime import datetime
import logging
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from blockflow.model import TrainingModel, EvaluationModel, ModelFactory, OthersWork
from .._utils.validation import validate_ndarray
from ..dp.dp_base import DPBase

from .dataset import LogisticRegressionDataset

class LogisticRegressionEvaluationModelParams(NamedTuple):
    output_classes: np.ndarray
    num_coefs: int
    validation_dataset: LogisticRegressionDataset
    sklearn_lr_kwargs: Optional[Dict[str, object]] = None

class LogisticRegressionEvaluationModelWeights(NamedTuple):
    estimator_coef: np.ndarray
    estimator_intercept: np.ndarray

class LogisticRegressionEvaluationModel(EvaluationModel):
    _logger = logging.getLogger(__name__)

    def __init__(self, params: LogisticRegressionEvaluationModelParams, weights: LogisticRegressionEvaluationModelWeights) -> None:
        sklearn_params = params.sklearn_lr_kwargs or {}
        if "warm_start" not in sklearn_params:
            sklearn_params["warm_start"] = True
        if "max_iter" not in sklearn_params:
            sklearn_params["max_iter"] = 25  # setting to something small to limit overfitting
        if "n_jobs" not in sklearn_params:
            sklearn_params["n_jobs"] = 1
        self._estimator = LogisticRegression(**sklearn_params)
        self._estimator.classes_ = params.output_classes
        self._validation_dataset = params.validation_dataset

        # if there's 2 classes then there's really only 1
        num_output_classes = len(self._estimator.classes_) if len(self._estimator.classes_) > 2 else 1

        validate_ndarray(weights.estimator_coef, (np.float, ), (num_output_classes, params.num_coefs), "estimator_coef")
        self._estimator.coef_ = weights.estimator_coef
        if self._estimator.fit_intercept:
            validate_ndarray(weights.estimator_intercept, (np.float, ), (num_output_classes, ), "estimator_intercept")
            self._estimator.intercept_ = weights.estimator_intercept

    def save(self, filename: str, mode: str = "xb") -> None:
        with open(filename, mode) as f:
            np.savez(f, estimator_coef=self._estimator.coef_, estimator_intercept=self._estimator.intercept_)

    def f1_score(self, dataset: Optional[LogisticRegressionDataset] = None) -> float:
        if dataset is None:
            dataset = self._validation_dataset
        ys = dataset.ys
        y_pred = np.expand_dims(self._estimator.predict(dataset.xs), axis=1)
        # # if len(self._estimator.classes_) == 2:
        #     average = 'binary'
        #     pos_label: object = self._estimator.classes_[1]
        # else:
        average = 'macro'
        pos_label = None
        f_1 = f1_score(ys, y_pred, average=average, pos_label=pos_label)
        assert isinstance(f_1, float)
        return f_1

    def score(self) -> float:
        return self.f1_score()

    def accuracy_score(self, dataset: Optional[LogisticRegressionDataset] = None) -> float:
        if dataset is None:
            dataset = self._validation_dataset
        ys = dataset.ys
        y_pred = np.expand_dims(self._estimator.predict(dataset.xs), axis=1)
        accuracy = accuracy_score(ys, y_pred)
        assert isinstance(accuracy, float)
        return accuracy

    def classification_report(self, dataset: Optional[LogisticRegressionDataset] = None) -> str:
        if dataset is None:
            dataset = self._validation_dataset
        ys = dataset.ys
        y_pred = np.expand_dims(self._estimator.predict(dataset.xs), axis=1)
        report = classification_report(ys, y_pred)
        assert isinstance(report, str)
        return report

    @property
    def estimator(self) -> LogisticRegression:
        return self._estimator

class LogisticRegressionTrainingModelParams(NamedTuple):
    training_dataset: LogisticRegressionDataset
    evaluation_params: LogisticRegressionEvaluationModelParams
    max_num_epochs: int = 100  # default to a large number
    checkpoint_folder: Optional[str] = None
    initial_weights: Optional[LogisticRegressionEvaluationModelWeights] = None
    dp: Optional[DPBase] = None  # defaults to no dp
    test_dataset: Optional[LogisticRegressionDataset] = None  # if you have ground truth data that you want evaluated against the shared model, specify it here

class LogisticRegressionTrainingModel(TrainingModel, LogisticRegressionEvaluationModel):
    def __init__(self, params: LogisticRegressionTrainingModelParams) -> None:
        self._training_dataset = params.training_dataset
        self._dp = params.dp if params.dp is not None else DPBase()
        self._max_num_epochs = params.max_num_epochs
        self._checkpoint_folder = params.checkpoint_folder
        # if there's 2 classes then there's really only 1
        num_output_classes = len(params.evaluation_params.output_classes) if len(params.evaluation_params.output_classes) > 2 else 1
        initial_weights = params.initial_weights if params.initial_weights is not None else LogisticRegressionEvaluationModelWeights(
            estimator_coef=np.random.random((num_output_classes, params.evaluation_params.num_coefs)),
            estimator_intercept=np.random.random((num_output_classes, )),
        )
        super().__init__(params.evaluation_params, initial_weights)
        self._class_to_indxs: Dict[object, np.ndarray] = {}
        self._class_to_indxs_orig: Dict[object, np.ndarray] = {}
        self._smallest_class_count: Optional[int] = None
        self._smallest_class: Optional[object] = None
        for class_ in self._estimator.classes_:
            class_indxs = np.argwhere(self._training_dataset.ys == class_)[:, 0]
            self._class_to_indxs[class_] = np.random.permutation(class_indxs)
            self._class_to_indxs_orig[class_] = class_indxs
            if (self._smallest_class_count is None) or (len(self._class_to_indxs[class_]) < self._smallest_class_count):
                self._smallest_class = class_
                self._smallest_class_count = len(self._class_to_indxs[class_])

    def train(self, dp_round: int, deadline: Optional[datetime] = None) -> None:
        best_score = self.score()
        best_params = self._estimator.coef_, self._estimator.intercept_
        epoch = 0
        batch_xs = self._training_dataset.xs
        batch_ys = self._training_dataset.ys
        while True:
            if epoch >= self._max_num_epochs:
                break
            if deadline is not None and datetime.now() >= deadline:
                break
            self._dp.perturb_batch_input(self._max_num_epochs, batch_xs)
            assert np.all(np.unique(batch_ys) == self._estimator.classes_), "don't have all the samples represented"
            self._estimator.fit(batch_xs, np.squeeze(batch_ys))
            assert np.all(np.unique(batch_ys) == self._estimator.classes_), "estimator classes changed"
            median_score = self.score()
            self._logger.debug("epoch %d f1_score %f accuracy_score %f n_iter %d", epoch, median_score, self.accuracy_score(), self._estimator.n_iter_)
            if median_score > best_score:
                self._logger.debug("epoch %d: score improved from %f to %f", epoch, best_score, median_score)
                best_score = median_score
                best_params = self._estimator.coef_, self._estimator.intercept_
            else:
                self._logger.debug("epoch %d: score decreased from %f to %f", epoch, best_score, median_score)
            if self._checkpoint_folder is not None:
                self.save(os.path.join(self._checkpoint_folder, f"dp_round_{dp_round}_epoch_{epoch}.dat"), "wb")
            epoch += 1
        self._estimator.coef_, self._estimator.intercept_ = best_params
        self._dp.perturb_round_weights(len(batch_ys), self._estimator.coef_)
        if self._estimator.fit_intercept:
            self._dp.perturb_round_weights(len(batch_ys), self._estimator.intercept_)

    def update(self, dp_round: int, others_work: Sequence[OthersWork]) -> None:
        # doing a weighted average, where each model is weighted in proportion to its score
        total_score = 0.0
        assert len(others_work) > 0, "cannot call on 0 models"
        sample_model = others_work[0].model
        assert isinstance(sample_model, LogisticRegressionEvaluationModel)
        estimator_coef = np.zeros_like(sample_model.estimator.coef_)
        estimator_intercept = np.zeros_like(sample_model.estimator.intercept_)
        for dummy_client, model, score in others_work:
            assert isinstance(model, LogisticRegressionEvaluationModel)
            total_score += score
            estimator_coef[...] = estimator_coef + model.estimator.coef_ * score
            estimator_intercept[...] = estimator_intercept + model.estimator.intercept_ * score
        self._estimator.coef_ = estimator_coef / total_score
        self._estimator.intercept_ = estimator_intercept / total_score
        if self._checkpoint_folder is not None:
            self.save(os.path.join(self._checkpoint_folder, f"dp_round_{dp_round}_merged.dat"))

class LogisticRegressionEvaluationModelFactory:
    def __init__(self, scoring_params: LogisticRegressionEvaluationModelParams):
        self._scoring_params = scoring_params

    def load(self, filename: str) -> LogisticRegressionEvaluationModel:
        with np.load(filename, "br") as data:
            weights = LogisticRegressionEvaluationModelWeights(
                estimator_coef=data['estimator_coef'],
                estimator_intercept=data['estimator_intercept'])
            model = LogisticRegressionEvaluationModel(self._scoring_params, weights)
            return model
            
class LogisticRegressionModelFactory(LogisticRegressionEvaluationModelFactory, ModelFactory):
    def __init__(self, training_params: LogisticRegressionTrainingModelParams, scoring_params: LogisticRegressionEvaluationModelParams):
        self._evaluation_params = training_params.evaluation_params
        self._model = LogisticRegressionTrainingModel(training_params)
        super().__init__(scoring_params)

    @property
    def training_model(self) -> LogisticRegressionTrainingModel:
        return self._model

    

