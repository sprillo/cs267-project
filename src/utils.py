from typing import List, Optional

import numpy as np


def quantize(
    branch_length: float, quantization_points: List[float]
) -> Optional[float]:
    if branch_length < min(quantization_points) or branch_length > max(
        quantization_points
    ):
        return None
    relative_errors = [
        max(abs(branch_length / q - 1), abs(q / branch_length - 1))
        for q in quantization_points
    ]
    argmin = np.argmin(relative_errors)
    return quantization_points[argmin]
