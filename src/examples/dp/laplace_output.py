import secrets

import numpy as np

from .dp_base import DPBase

class LaplaceOutputDP(DPBase):
    def __init__(self, *, inv_reg_term: float, epsilon: float):
        self._inv_reg_term = inv_reg_term
        self._epsilon = epsilon

    def perturb_round_weights(self, num_unique_records: int, weights: np.ndarray) -> None:
        scale = 2.0 * self._inv_reg_term / (num_unique_records * self._epsilon)
        random_seed = secrets.randbits(32)
        np_random = np.random.RandomState(random_seed)
        weights += np_random.laplace(loc=0.0, scale=scale, size=weights.shape)
