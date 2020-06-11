import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

class LogisticRegressionDataset:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self._xs = xs
        self._ys = ys

    def __len__(self) -> int:
        return len(self._xs)

    @property
    def xs(self) -> np.ndarray:
        return self._xs

    @property
    def ys(self) -> np.ndarray:
        return self._ys

    def save(self, filename: str) -> None:
        with open(filename, "wb+") as f:
            np.savez(f, xs=self._xs, ys=self._ys)
    
    @classmethod
    def load(cls, filename: str) -> 'LogisticRegressionDataset':
        with np.load(filename, "rb") as f:
            return cls(f['xs'], f['ys'])
