from typing import Tuple, Sequence

import numpy as np

def validate_ndarray(data: np.ndarray, dtypes: Sequence[object], shape: Tuple[int, ...], array_name_for_error: str = "") -> None:
    if not isinstance(data, np.ndarray):
        raise Exception(f"{array_name_for_error} is not an np.ndarray")
    if not data.shape == shape:
        raise Exception(f"{array_name_for_error}.shape({data.shape}) != {shape}")
    if not data.dtype in dtypes:
        raise Exception(f"{array_name_for_error}.dtype({data.dtype}) not in {dtypes}")
