"""
matcher.py — Compare a query descriptor against a list of template descriptors.

Callers own storage. Templates are passed in as plain tuples; the caller
fetches them from whatever DB or list they maintain.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import descriptor_distance_cyclic, DEFAULT_N, OFFSETS


@dataclass
class MatchResult:
    id: Any
    distance: float


def match(
    query_desc: np.ndarray,
    templates: list[tuple[Any, np.ndarray, np.ndarray]],
    n_points: int = DEFAULT_N,
    n_offsets: int = len(OFFSETS),
    top_k: int | None = None,
) -> list[MatchResult]:
    """
    Rank templates by shape similarity to the query.

    Parameters
    ----------
    query_desc : np.ndarray
        Descriptor for the query shape, as returned by build_descriptor().
    templates : list of (id, descriptor, reversed_descriptor)
        Each entry is a (id, fwd_desc, rev_desc) tuple. Both descriptors must
        have been built with the same n_points / n_offsets as the query.
        Passing both directions avoids recomputing the reversal per query.
    n_points : int
        Number of contour sample points used when building the descriptors.
    n_offsets : int
        Number of offset values used when building the descriptors.
    top_k : int or None
        If set, return only the top_k closest matches.

    Returns
    -------
    list[MatchResult]
        Results sorted by distance ascending (closest first).
    """
    results = []
    for id_, fwd, rev in templates:
        d_fwd = descriptor_distance_cyclic(query_desc, fwd, n_points, n_offsets)
        d_rev = descriptor_distance_cyclic(query_desc, rev, n_points, n_offsets)
        results.append(MatchResult(id=id_, distance=min(d_fwd, d_rev)))
    results.sort(key=lambda r: r.distance)
    if top_k is not None:
        results = results[:top_k]
    return results
