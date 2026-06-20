from .image_processing import (
    load_image,
    load_image_bytes,
    extract_main_contour,
    remove_background,
    ContourResult,
    ContourExtractionError,
)
from .geometry import (
    sample_contour_uniform,
    build_descriptor,
    reverse_sampled,
    descriptor_distance_cyclic,
    DEFAULT_N,
    OFFSETS,
)
from .matcher import match, MatchResult
