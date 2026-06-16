"""Drawing utilities for ML inference visualization."""

from collections.abc import Sequence
from enum import IntEnum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .parse import Classification, Detection, Mask, Pose


BLACK = (0, 0, 0)

# Pascal VOC colormap (21 classes)
COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
PADDING = 10

# COCO keypoint format (17 keypoints):
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

WHITE = (255, 255, 255)


class XAlign(IntEnum):
    """Horizontal alignment options for text rendering."""

    LEFT = 0
    CENTER = 1
    RIGHT = 2


class YAlign(IntEnum):
    """Vertical alignment options for text rendering."""

    BOTTOM = 0
    MIDDLE = 1
    TOP = 2


def _draw_contours(
    img: np.ndarray,
    contours: Sequence[np.ndarray],
    color: Sequence[float] | float = WHITE,
    thickness: int = 2,
) -> None:
    cv2.drawContours(img, contours, -1, color, thickness)


def _draw_lines(
    img: np.ndarray,
    pts: Sequence[np.ndarray],
    *,
    is_closed: bool = False,
    color: Sequence[float] | float = WHITE,
    thickness: int = 2,
) -> None:
    cv2.polylines(img, pts, is_closed, color, thickness)


def _draw_progress(img: np.ndarray, progress: float) -> None:
    bar_ratio = 0.33
    bar_h = PADDING
    bar_y_gap = 2 * bar_h

    img_h, img_w = img.shape[:2]
    bar_w = int(img_w * bar_ratio)
    bar_x = (img_w - bar_w) // 2
    bar_y = img_h // 2 + bar_y_gap

    bar_end = (bar_x + bar_w, bar_y + bar_h)
    _draw_rectangle(img, (bar_x, bar_y), bar_end, WHITE)
    fill_end = (bar_x + int(bar_w * progress), bar_y + bar_h)
    _draw_rectangle(img, (bar_x, bar_y), fill_end, WHITE, thickness=-1)


def _draw_rectangle(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: Sequence[float] | float = WHITE,
    thickness: int = 1,
) -> None:
    cv2.rectangle(img, pt1, pt2, color, thickness)


def _draw_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int = FONT,
    font_scale: float = FONT_SCALE,
) -> None:
    cv2.putText(img, text, pos, font, font_scale, BLACK, 2)
    cv2.putText(img, text, pos, font, font_scale, WHITE, 1)


def _draw_text_aligned(
    img: np.ndarray,
    text: str,
    x_align: XAlign = XAlign.CENTER,
    y_align: YAlign = YAlign.MIDDLE,
) -> None:
    text_w, text_h = _get_text_size(text)
    img_h, img_w = img.shape[:2]

    if y_align == YAlign.TOP:
        y = text_h + PADDING
    elif y_align == YAlign.MIDDLE:
        y = img_h // 2 + text_h // 2
    else:
        y = img_h - PADDING

    if x_align == XAlign.LEFT:
        x = PADDING
    elif x_align == XAlign.CENTER:
        x = img_w // 2 - text_w // 2
    else:
        x = img_w - text_w - PADDING

    _draw_text(img, text, (x, y))


def _find_centroid(contours: Sequence[np.ndarray]) -> tuple[int, int] | None:
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    return None


def _find_contours(mask: np.ndarray) -> Sequence[np.ndarray]:
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def _get_text_size(
    text: str,
    font: int = FONT,
    font_scale: float = FONT_SCALE,
) -> tuple[int, int]:
    w, h = cv2.getTextSize(text, font, font_scale, 1)[0]
    return (w, h)


def draw_classifications(img: np.ndarray, results: list["Classification"]) -> None:
    """Draw classification labels and scores onto the image."""
    for index, result in enumerate(results):
        first_label = result.label.split(":", 1)[-1].split(",")[0].strip()
        text = f"{first_label} {result.score * 100:.0f}%"
        _, text_h = _get_text_size(text)
        pos = (PADDING, (text_h + PADDING) * (index + 1))

        _draw_text(img, text, pos)


def draw_detections(img: np.ndarray, results: list["Detection"]) -> None:
    """Draw bounding boxes and labels for object detections onto the image."""
    for detection in results:
        box_x, box_y, box_w, box_h = detection.box
        text = f"{detection.label} {detection.score * 100:.0f}%"

        _draw_text(img, text, (box_x + PADDING // 2, int(box_y + PADDING * 1.5)))
        _draw_rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h))


def draw_masks(img: np.ndarray, results: list["Mask"]) -> None:
    """Draw segmentation mask contours and labels onto the image."""
    for mask in results:
        contours = _find_contours(mask.binary_mask)
        if not contours:
            continue

        _draw_contours(img, contours, COLORS[mask.class_id % len(COLORS)])

        centroid = _find_centroid(contours)
        if centroid:
            _draw_text(img, mask.label, centroid)


def draw_message(img: np.ndarray, text: str, progress: float = 0) -> None:
    """Draw a centered message and optional progress bar onto the image."""
    _draw_text_aligned(img, text)
    _draw_progress(img, progress)


def draw_overlay(img: np.ndarray, model_name: str, fps: int = 0) -> None:
    """Draw model name and FPS counter onto the bottom corners of the image."""
    _draw_text_aligned(img, model_name, XAlign.LEFT, YAlign.BOTTOM)
    _draw_text_aligned(img, f"{fps} fps", XAlign.RIGHT, YAlign.BOTTOM)


def draw_poses(img: np.ndarray, results: list["Pose"]) -> None:
    """Draw skeleton keypoint connections for pose estimations onto the image."""
    segments = []
    for pose in results:
        for start_idx, end_idx in SKELETON:
            start_kp = pose.keypoints[start_idx]
            end_kp = pose.keypoints[end_idx]
            if start_kp[2] > 0 and end_kp[2] > 0:
                pt1 = int(start_kp[0]), int(start_kp[1])
                pt2 = int(end_kp[0]), int(end_kp[1])
                segments.append(np.array([pt1, pt2], dtype=np.int32))
    if segments:
        _draw_lines(img, segments)


def save_image(img: np.ndarray, filename: str) -> None:
    """Save an RGB image to disk as BGR."""
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
