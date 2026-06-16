"""Parsing utilities for IMX500 inference outputs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Classification:
    """A single image classification result."""

    score: float
    label: str


@dataclass
class Detection:
    """A single object detection result with bounding box."""

    box: tuple[int, int, int, int]
    score: float
    label: str


@dataclass(eq=False)
class Mask:
    """A single segmentation mask result."""

    class_id: int
    label: str
    binary_mask: np.ndarray


@dataclass(eq=False)
class Pose:
    """A single pose estimation result with keypoints."""

    keypoints: np.ndarray
    score: float


def _convert_coords(
    coords: tuple[float, float, float, float],
    display_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = coords
    display_w, display_h = display_size

    x = int(max(0, x0) * display_w)
    y = int(max(0, y0) * display_h)
    w = int((x1 - x0) * display_w)
    h = int((y1 - y0) * display_h)

    return (x, y, w, h)


def _resize_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape
    new_w, new_h = size
    rows = np.arange(new_h) * h // new_h
    cols = np.arange(new_w) * w // new_w
    return arr[rows[:, None], cols]


def _softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return y / np.expand_dims(np.sum(y, axis=-1), axis=-1)


def parse_classifications(
    outputs: list[np.ndarray],
    labels: list[str],
    threshold: float = 0.05,
    *,
    softmax: bool = False,
) -> list[Classification]:
    """Parse model outputs into top classification results above threshold.

    Returns:
        Classifications above threshold, sorted by score descending.

    """
    output = outputs[0][0]

    if softmax:
        output = _softmax(output)

    above_threshold = np.where(output > threshold)[0]
    sorted_indices = above_threshold[np.argsort(-output[above_threshold])][:3]

    return [Classification(float(output[idx]), labels[idx]) for idx in sorted_indices]


def parse_detections(
    outputs: list[np.ndarray],
    labels: list[str],
    display_size: tuple[int, int],
    input_size: tuple[int, int],
    threshold: float = 0.55,
    *,
    bbox_normalization: bool = False,
    bbox_order: str = "yx",
) -> list[Detection]:
    """Parse model outputs into object detections above threshold.

    Returns:
        Detections with bounding boxes scaled to display size.

    """
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]

    if bbox_normalization:
        boxes /= input_size[1]

    if bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]  # convert to yx order

    return [
        Detection(
            _convert_coords(tuple(box), display_size),
            score,
            labels[int(cls)],
        )
        for box, score, cls in zip(boxes, scores, classes, strict=True)
        if score > threshold
    ]


def parse_masks(
    outputs: list[np.ndarray],
    labels: list[str],
    display_size: tuple[int, int],
    threshold: float = 0.1,
) -> list[Mask]:
    """Parse model outputs into segmentation masks above pixel-coverage threshold.

    Returns:
        Masks for classes occupying at least threshold fraction of the display.

    """
    mask_data = _resize_nearest(outputs[0][0], display_size)

    # Count pixels per class
    counts = np.bincount(mask_data.ravel().astype(np.int32), minlength=len(labels))

    min_pixels = threshold * display_size[0] * display_size[1]
    found_indices = np.nonzero(counts >= min_pixels)[0]
    found_indices = found_indices[found_indices != 0]  # drop background

    return [
        Mask(int(i), labels[int(i)], (mask_data == i).astype(np.uint8))
        for i in found_indices
    ]


def parse_poses(
    outputs: list[np.ndarray],
    display_size: tuple[int, int],
    threshold: float = 0.3,
) -> list[Pose]:
    """Parse model outputs into pose estimations above keypoint threshold.

    Returns:
        Poses with keypoints scaled to display size.

    """
    from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

    keypoints, scores, _ = postprocess_higherhrnet(
        outputs=outputs,
        img_size=(display_size[1], display_size[0]),
        img_w_pad=(0, 0),
        img_h_pad=(0, 0),
        network_postprocess=True,
        detection_threshold=threshold,
        max_num_people=1,
    )

    if not scores:
        return []

    # Reshape from flat (num_people, 51) to (num_people, 17 joints, [x, y, confidence])
    keypoints_array = np.stack(keypoints).reshape(-1, 17, 3)

    return [Pose(kp, score) for kp, score in zip(keypoints_array, scores, strict=True)]
