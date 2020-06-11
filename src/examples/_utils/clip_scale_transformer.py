from typing import Optional

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
import numpy as np

class ClipScaleTransformer(TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(self, min_value: float, max_value: float, copy: bool = False) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.copy = copy

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> 'ClipScaleTransformer':  # pylint: disable=unused-argument
        # since the min and max are predetermined, there is no "fitting" to do
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = check_array(x, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")
        np.clip(x, self.min_value, self.max_value, out=x)
        x -= self.min_value
        x /= (self.max_value - self.min_value)
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        # Note that the inverse is approximate if the original X was outside of the range
        x = check_array(x, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")
        x *= (self.max_value - self.min_value)
        x += self.min_value
        return x
