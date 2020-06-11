from typing import List

import numpy as np

class InvalidShapeError(Exception):
    pass

class InvalidDTypeError(Exception):
    pass

class InvalidTypeError(Exception):
    pass

class OtherValidationError(Exception):
    pass

class MyDataset:
    def __init__(self, x_s: np.ndarray, y_s: np.ndarray) -> None:
        self._x_s, self._y_s = x_s, y_s
        if not isinstance(self._x_s, np.ndarray):
            raise InvalidTypeError()
        if not isinstance(self._y_s, np.ndarray):
            raise InvalidTypeError()
        n = len(self._x_s)
        if not self._x_s.shape == (n, 2):
            raise InvalidShapeError()
        if not self._y_s.shape == (n, 1):
            raise InvalidShapeError()
        if not self._x_s.dtype == np.float:
            raise InvalidDTypeError()
        if not self._y_s.dtype == np.float:
            raise InvalidDTypeError()

    def __len__(self) -> int:
        return len(self._x_s)

    @property
    def x_s(self) -> np.ndarray:
        return self._x_s

    @property
    def y_s(self) -> np.ndarray:
        return self._y_s

    def save(self, filename: str) -> None:
        np.savez(filename, x_s=self._x_s, y_s=self._y_s)

class MyDatasetFactory:
    def __init__(self, correct_factor: float = 1.0, noise_factor: float = 0.0, offset: float = 0.0) -> None:
        self._noise_factor = noise_factor
        self._correct_factor = correct_factor
        self._offset = offset

    def generate(self, num_points: int) -> MyDataset:
        # Let the function be y = 2x[0] + 3x[1]
        x_s = np.random.random((num_points, 2)) - 0.5  # centering around 0
        y_s = np.expand_dims(2 * x_s[:, 0] + 3 * x_s[:, 1], axis=1)
        y_s_noisy = self._correct_factor * y_s + self._noise_factor * np.random.normal(size=y_s.shape) + self._offset
        return MyDataset(x_s, y_s_noisy)

    @staticmethod
    def load(filename: str) -> MyDataset:
        with np.load(filename) as data:
            return MyDataset(x_s=data['x_s'], y_s=data['y_s'])

    @staticmethod
    def join(*datasets: MyDataset) -> MyDataset:
        x_s: List[np.ndarray] = []
        y_s: List[np.ndarray] = []
        for dataset in datasets:
            x_s.append(dataset.x_s)
            y_s.append(dataset.y_s)
        return MyDataset(x_s=np.vstack(tuple(x_s)), y_s=np.vstack(tuple(y_s)))
