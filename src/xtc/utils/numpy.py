#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Sequence
import numpy as np
import numpy.typing

from .math import mulall


def np_init(shape: Sequence[int], dtype: str) -> numpy.typing.NDArray[Any]:
    """
    Initialize and return a NP array filled with the 17 evenly
    spaced values in [-1, 1] with step 1/8: -1, -0.875, ..., 0, ..., 1.
    Values are exact in binary floating point.
    """
    vals = np.arange(mulall(list(shape)))
    return ((vals % 17 - 8) / 8).reshape(shape).astype(dtype)
