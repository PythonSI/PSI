import numpy as np
import numpy.typing as npt 
from typing import Tuple
from mpmath import mp 

mp.dps = 500

def FormatArray(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def intersect(interval1, interval2):
    """
    Intersect two intervals [low, high], where None means unbounded.
    Returns [low, high] or [] if no overlap.
    """
    l1, u1 = interval1
    l2, u2 = interval2

    # Handle lower bounds
    if l1 is None:
        lower = l2
    elif l2 is None:
        lower = l1
    else:
        lower = max(l1, l2)

    # Handle upper bounds
    if u1 is None:
        upper = u2
    elif u2 is None:
        upper = u1
    else:
        upper = min(u1, u2)

    # Check feasibility
    if (lower is not None) and (upper is not None) and (lower > upper):
        assert False, "Logic error in intersect_intervals"

    return [lower, upper]

def solve_linear_inequalities(A: np.ndarray, B: np.ndarray):
    """
    Vectorized solver for global inequality A + Bz <= 0.
    Returns a single interval [lower, upper], where None means unbounded.
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    
    if A.size == 0:
        return [-np.inf, np.inf]

    # Case 1: B > 0  →  z <= -A/B
    mask_pos = B > 1e-10
    upper_bounds = np.full(A.shape, np.inf, dtype=float)
    upper_bounds[mask_pos] = -A[mask_pos] / B[mask_pos]

    # Case 2: B < 0  →  z >= -A/B
    mask_neg = B < -1e-10
    lower_bounds = np.full(A.shape, -np.inf, dtype=float)
    lower_bounds[mask_neg] = -A[mask_neg] / B[mask_neg]

    # Case 3: B == 0
    mask_zero = (B >= -1e-10) & (B <= 1e-10)
    if np.any(mask_zero & (A > 0)):
        assert False, "No satisfying solution"

    # Global bounds
    lower = np.max(lower_bounds)
    upper = np.min(upper_bounds)

    # Convert infinities to None for readability
    low = None if np.isneginf(lower) else lower
    up  = None if np.isposinf(upper) else upper

    # Check feasibility
    if (low is not None) and (up is not None) and (low > up):
        assert False, "Logic error in solve_linear_inequalities"

    return [low, up]

import numpy as np

def solve_quadratic_inequality(a: float, b: float, c: float, z: float, tol: float = 1e-12):
    """
    Solve quadratic inequality: a z^2 + b z + c <= 0.
    Returns the single interval [low, high] that contains z.
    Uses -np.inf / np.inf for unbounded sides.
    Asserts if no interval contains z.
    """
    # Linear case
    if abs(a) < tol:
        if abs(b) < tol:
            interval = [] if c > tol else [-np.inf, np.inf]
        elif b > 0:
            interval = [-np.inf, -c / b]
        else:
            interval = [-c / b, np.inf]
    else:
        # Quadratic case
        D = b**2 - 4*a*c
        if D < -tol:
            interval = [] if a > 0 else [-np.inf, np.inf]
        elif abs(D) <= tol:
            r = -b / (2*a)
            if a > 0:
                interval = [r, r] if abs(z - r) <= tol else []
            else:
                interval = [-np.inf, np.inf]
        else:  # D > 0
            sqrtD = np.sqrt(D)
            r1, r2 = sorted([(-b - sqrtD) / (2*a), (-b + sqrtD) / (2*a)])
            if a > 0:
                interval = [r1, r2] if r1 <= z <= r2 else []
            else:
                left = [-np.inf, r1]
                right = [r2, np.inf]
                if left[0] <= z <= left[1]:
                    interval = left
                elif right[0] <= z <= right[1]:
                    interval = right
                else:
                    interval = []

    # Assert if no solution
    assert interval != [], f"No solution interval contains z={z}"
    return interval


def compute_p_value(test_statistic, variance, list_intervals, list_outputs, observed_output):
    mp.dps = 500
    numerator = 0
    denominator = 0
    
    standard_deviation = np.sqrt(variance)
    for i in range(len(list_intervals)):
        left, right = list_intervals[i]
        output = list_outputs[i]

        if (np.array_equal(output, observed_output) == False):
            continue
        
        denominator = denominator + mp.ncdf(right / standard_deviation) - mp.ncdf(left / standard_deviation)
        if test_statistic >= right:
            numerator = numerator + mp.ncdf(right /standard_deviation) - mp.ncdf(left / standard_deviation)
        elif (test_statistic >= left) and (test_statistic< right):
            numerator = numerator + mp.ncdf(test_statistic / standard_deviation) - mp.ncdf(left / standard_deviation)
    
    cdf = float(numerator / denominator)
    return 2 * min(cdf, 1 - cdf)