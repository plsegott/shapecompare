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


def sample_contour_uniform(pts: np.ndarray, n: int = DEFAULT_N) -> np.ndarray:
    """
    Resample an ordered contour to n points uniformly spaced by arc length.

    Parameters
    ----------
    pts : np.ndarray
        An (M, 2) array of (x, y) coordinates representing the contour points.
    n : int, optional
        The number of points to sample along the contour (default is 64).

    Returns
    -------
    np.ndarray
        An (n, 2) array of (x, y) coordinates, uniformly sampled by arc length.
    """
    
    if len(pts) < 2:
        raise ValueError("At least 2 points are required to sample a contour.")
    
    #Close the contour by appending the first point to the end
    closed = np.vstack([pts, pts[:1]]) # (M+1, 2)
    
    #Segment lengths between consecutive points
    diffs = np.diff(closed, axis=0) # (M, 2)
    seg_lengths = np.linalg.norm(diffs, axis=1) # (M,)
    
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)]) # (M+1,)
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
        
    # Stack offsets as columns: shape (n, len(offsets)), then flatten row-major.
    matrix = np.column_stack(rows)  # (n, len(offsets))
    return matrix.ravel().astype(np.float32)

def reverse_sampled(sampled: np.ndarray) -> np.ndarray:
    """
    Reverse the point ordering of a sampled contour.

    Comparing a query against both the forward and reversed template
    handles the case where OpenCV traces the boundary in a different
    direction (CW vs CCW) between images.
    """
    return sampled[::-1].copy()

def descriptor_distance_cyclic(
    query_desc: np.ndarray,
    template_desc: np.ndarray,
    n_points: int,
    n_offsets: int,
    use_reversal: bool = True,
) -> float:
    """
    Compute the minimum L2 distance between two descriptors under all cyclic shifts.
    use_reversal is accepted for API compatibility but reversal must be handled by
    passing the pre-computed reversed descriptor from the DB.
    """
    q = query_desc.reshape(n_points, n_offsets)
    t = template_desc.reshape(n_points, n_offsets)
    return float(_min_cyclic_distance(q, t))

def _min_cyclic_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Minimum L2 distance between two (n, k) descriptor matrices under all n cyclic shifts.
    """
    n = len(a)
    # Stack all cyclic shifts of b into (n, n, k) and compare against a (n, k).
    # More memory-efficient: iterate shifts but keep it in numpy.
    min_dist = np.inf
    for shift in range(n):
        shifted = np.roll(b, shift, axis=0)
        diff = a - shifted
        dist = np.sqrt(np.sum(diff * diff))
        if dist < min_dist:
            min_dist = dist
    return min_dist