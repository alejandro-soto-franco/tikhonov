"""Type stubs for the `tikhonov._core` extension module."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__version__: str

class HarmonyResult:
    Z_corr: NDArray[np.float64]
    Y: NDArray[np.float64]
    R: NDArray[np.float64]
    history: list[dict[str, Any]]
    converged: bool
    n_iter: int

def py_run_harmony(
    z: NDArray[np.float64],
    labels: NDArray[np.uint32],
    **kwargs: Any,
) -> HarmonyResult: ...
