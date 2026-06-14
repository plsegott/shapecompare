"""
geometry.py — Uniform arc-length contour sampling and descriptor generation.

Core algorithm
--------------
1. Given an ordered sequence of boundary points from OpenCV, compute the
   cumulative arc length along the contour (treating it as closed).
2. Sample N points at equal intervals in arc-length space (linear interpolation).
3. For each sampled point i and each offset d in OFFSETS:
       dist(i, d) = ||p[i] - p[(i+d) % N]||₂  / perimeter
   This gives a (N × len(OFFSETS)) matrix — flattened to length N*len(OFFSETS).
4. The descriptor is invariant to translation and scale (via perimeter normalisation).
   Rotation tolerance is handled during comparison via cyclic shifts.
"""

import numpy as np


DEFAULT_N = 64

OFFSETS = [2, 4, 8, 16, 32]


def sample_contour_unifor(pts: np.ndarray, n: int = DEFAULT_N) -> np.ndarray:
    """
    
    Resample and ordered contour points to a fixed number of points, uniformly spaced along the contour.
    
    Parameters
    ----------
    pts : np.ndarray
        An (M, 2) array of (x, y) coordinates representing the contour points.
    n : int, optional
        The number of points to sample along the contour (default is 64).
    
    Returns
    -------
    np.ndarray
        An (n, 2) array of (x, y) coordinates representing the uniformly sampled contour points.
    
    Uniformly sample n points along the contour defined by pts."""
    
    if len(pts) < 2:
        raise ValueError("At least 2 points are required to sample a contour.")
    
    #Close the contour by appending the first point to the end
    closed = np.vstack([pts, pts[:1]]) # (M+1, 2)
    
    #Segment lengths between consecutive points
    diffs = np.diff(closed, axis=0) # (M, 2)
    seg_lengths = np.linalg.norm(diffs, axis=1) # (M,)
    
    cumlen = np.concatenate([0.0], np.cumsum(seg_lengths)) # (M+1,)
    total = cumlen[-1]
    
    if total < 1e-9:
        raise ValueError("Contour has near-zero perimeter — cannot sample.")
    
    targets = np.linspace(0, total, n, endpoint=False) # (n,)
    
    sampled_x = np.interp(targets, cumlen, closed[:, 0])
    sampled_y = np.interp(targets, cumlen, closed[:, 1])
    
    return np.stack([sampled_x, sampled_y], axis=1).astype(np.float32) # (n, 2)

def build_descriptor(sampled: np.ndarray, perimeter:float, offsets: list[int] = OFFSETS) -> np.ndarray:
    """
    Build a shape descriptor from uniformly sampled contour points.
    
    Parameters
    ----------
    sampled : np.ndarray
        An (N, 2) array of (x, y) coordinates representing the uniformly sampled contour points.
    perimeter : float
        The total perimeter length of the contour, used for normalisation.
    offset : list[int], optional
        A list of integer offsets to compute pairwise distances (default is [2, 4, 8, 16, 32]).
    
    Returns
    -------
    np.ndarray
        A 1D array of length N * len(offset) representing the shape descriptor.
    
    For each point i and each offset d in offset, compute the normalised distance:
        dist(i, d) = ||p[i] - p[(i+d) % N]||₂ / perimeter
    This results in a descriptor that is invariant to translation and scale."""
    
    if perimeter < 1e-9:
        raise ValueError("Perimeter is near zero — cannot normalise descriptor.")
    
    n = len(sampled)
    rows = []
    for d in offsets:
        # Roll the array by d positions and compute pointwise distances
        rolled = np.roll(sampled, -d, axis=0)  # (N, 2)
        dists = np.linalg.norm(sampled-rolled, axis=1) # (N,)
        rows.append(dists/perimeter)