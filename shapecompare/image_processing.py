"""
image_processing.py — Image loading, thresholding, morphology, contour extraction.

Handles three input types automatically:
  1. Dark hanger on white/light background (most template line drawings, white-paper photos)
  2. Light/silver hanger on dark/black background (product photos on black)
  3. Outline drawings (thin line art) — interior is flood-filled to make solid
"""

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class ContourResult:
    contour: np.ndarray  #shape (N, 2), dtype np.float32
    binary: np.ndarray # cleaned binary mask (uint8, 0/255)
    gray: np.ndarray      # grayscale source image
    original: np.ndarray  # original BGR image
    perimeter: float
    area: float

class ContourExtractionError(Exception):
    pass

def load_image(path: str ) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ContourExtractionError(f"Failed to load image: {path}")
    return img

def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ContourExtractionError("Failed to decode image from bytes")
    return img

def _corner_brightness(img: np.ndarray, fraction: float = 0.05) -> float:
    
    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    ch = max(1, int(h* fraction))
    cw = max(1, int(w* fraction))
    corners = np.concatenate([
        gray[:ch, :cw].ravel(),
        gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(),
        gray[-ch:, -cw:].ravel()
    ])
    return float(corners.mean())

def _flood_fill_interior(binary: np.ndarray) -> np.ndarray:
    """
    Fill the interior of a thin outline drawing to make it a solid shape.
    Flood-fills from the border (guaranteed background), then inverts.
    """
    h, w = binary.shape
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood = flood[1:h+1, 1:w+1]
    filled = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, filled)

def _pick_better_binary(b1: np.ndarray, b2: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Given two candidate binary masks, return the one whose largest contour
    is more central and compact (less likely to be the image border).
    """
    h, w = shape

    def score(b: np.ndarray) -> float:
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return -1.0
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # Penalise contours that are close to the full image size (= border artifact)
        if area > 0.85 * h * w:
            return -1.0
        # Reward larger, more central blobs
        M = cv2.moments(c)
        if M["m00"] == 0:
            return area
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist_from_centre = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
        return area / (1.0 + dist_from_centre)

    return b1 if score(b1) >= score(b2) else b2

def extract_main_contour(
    img: np.ndarray,
    morph_kernel_size: int = 3,
    morph_iterations: int = 1,
) -> ContourResult:
    """
    Full pipeline: BGR image → largest external contour.

    Auto-detects background type from corner brightness:
      - Light background (>180): invert so dark object becomes white
      - Dark background (<80): no invert, bright object stays white
      - Mid-tone background: try both, pick the one with the larger clean contour
    Also detects outline drawings and flood-fills them.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = _corner_brightness(img)

    # CLAHE boosts local contrast so metallic/silver objects near white backgrounds
    # become distinguishable even when highlights are nearly white.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if brightness > 180:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    elif brightness < 80:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, b_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, b_nrm = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = _pick_better_binary(b_inv, b_nrm, img.shape[:2])

    # Dilate to bridge gaps from bright highlights on metallic surfaces.
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, dilate_k, iterations=2)

    # Outline drawing detection: thin outline → flood-fill interior.
    fill_ratio = np.count_nonzero(binary) / max(binary.size, 1)
    if fill_ratio < 0.15:
        binary = _flood_fill_interior(binary)

    # Large closing kernel to fill holes left by highlights/shadows inside hanger.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ContourExtractionError("No contours found. Check image quality.")

    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < 100:
        raise ContourExtractionError(
            f"Largest contour area ({area:.1f} px²) is too small."
        )

    perimeter = cv2.arcLength(main, closed=True)
    if perimeter < 10:
        raise ContourExtractionError("Contour perimeter is too small.")

    # Draw the contour filled on a blank mask — this eliminates any holes caused
    # by internal shading or shadows that fooled the threshold step.
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, [main], -1, 255, thickness=cv2.FILLED)

    pts = main.reshape(-1, 2).astype(np.float32)
    return ContourResult(
        contour=pts,
        binary=filled,
        gray=gray,
        original=img,
        perimeter=float(perimeter),
        area=float(area),
    )