"""Hardware interfaces for display, camera, and battery."""

import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import smbus2

from displayhatmini import DisplayHATMini

from .draw import (
    draw_classifications,
    draw_detections,
    draw_masks,
    draw_message,
    draw_overlay,
    draw_poses,
)
from .parse import (
    parse_classifications,
    parse_detections,
    parse_masks,
    parse_poses,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an IMX500 model."""

    name: str = "inputtensoronly"
    task: str = ""
    bbox_normalization: bool = False
    bbox_order: str = "yx"
    ignore_dash_labels: bool = False
    labels: str | None = None
    softmax: bool = False


class Display:
    """Hardware interface for the DisplayHATMini screen and buttons."""

    BUTTON_A = DisplayHATMini.BUTTON_A
    BUTTON_B = DisplayHATMini.BUTTON_B
    BUTTON_X = DisplayHATMini.BUTTON_X
    BUTTON_Y = DisplayHATMini.BUTTON_Y

    def __init__(
        self,
        size: tuple[int, int] = (DisplayHATMini.WIDTH, DisplayHATMini.HEIGHT),
    ) -> None:
        """Initialise the display buffer at the given pixel dimensions."""
        width, height = size
        self._buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self._display: DisplayHATMini

    def start(self) -> None:
        """Initialise the DisplayHATMini and turn on the backlight."""
        logger.info("Starting display...")
        try:
            self._display = DisplayHATMini(self._buffer)
            self._display.display()
            self._display.set_backlight(1.0)
            self._display.set_led(0.05, 0.05, 0.05)
            logger.info("Started display")
        except Exception:
            logger.exception("Failed to start display")
            raise

    def get_buffer(self) -> np.ndarray:
        """Return the shared numpy frame buffer.

        Returns:
            The display buffer array.

        """
        return self._buffer

    def set_buffer(self, frame: np.ndarray) -> None:
        """Copy frame into the display buffer in-place."""
        self._buffer[:] = frame

    def show(self) -> None:
        """Push the current buffer to the screen."""
        self._display.display()

    def message(self, text: str, progress: float = 0, *, show: bool = True) -> None:
        """Clear the buffer, draw a status message, and optionally show it."""
        self._buffer[:] = 0
        draw_message(self._buffer, text, progress)
        if show:
            self.show()

    def on_button_pressed(self, callback: Callable[[int], None]) -> None:
        """Register a callback to fire when a button is pressed."""
        self._display.on_button_pressed(callback)

    def read_button(self, pin: int) -> bool:
        """Return True if the button at the given pin is currently pressed.

        Returns:
            Button state.

        """
        return self._display.read_button(pin)

    def set_backlight(self, brightness: float) -> None:
        """Set the backlight brightness from 0.0 (off) to 1.0 (full)."""
        self._display.set_backlight(brightness)

    def set_led(self, r: float, g: float, b: float) -> None:
        """Set the RGB LED colour with each channel from 0.0 to 1.0."""
        self._display.set_led(r, g, b)

    def stop(self) -> None:
        """Clear the screen, turn off the backlight, and release the display."""
        logger.info("Stopping display...")
        try:
            if not hasattr(self, "_display"):
                return
            self._clear()
            self._display.display()
            self._display.set_backlight(0)
            self._display.set_led(0, 0, 0)
            logger.info("Stopped display")
        except Exception:
            logger.exception("Failed to stop display")
            raise

    def _clear(self) -> None:
        self._buffer[:] = 0


class Camera:
    """Manages the IMX500 camera, model loading, inference, and frame callbacks."""

    MODELS_DIR = Path("/usr/share/imx500-models")

    def __init__(
        self,
        display_size: tuple[int, int] = (DisplayHATMini.WIDTH, DisplayHATMini.HEIGHT),
    ) -> None:
        """Initialise the camera with paths and target display size."""
        self.fps: int = 0
        self.is_switching: bool = False
        self.results: list = []
        self._display_size = display_size
        self._model_name: str = ""
        self._overlay_visible: bool = False
        self._show_overlay_until: float = 0

    def load_model(self, model_config: ModelConfig) -> None:
        """Load an IMX500 model and configure task, labels, and intrinsics."""
        self._model_name = model_config.name
        logger.info("Loading model: %s...", self._model_name)
        try:
            from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

            name = self._model_name.removeprefix("imx500_network_")
            model_file = f"imx500_network_{name}.rpk"
            model_path = self.MODELS_DIR / model_file
            if not model_path.exists():
                model_path = files("picam") / "models" / model_file
            self._imx500 = IMX500(str(model_path))
            intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
            self._intrinsics = self._configure_intrinsics(intrinsics, model_config)
            self._input_size = self._imx500.get_input_size()
        except Exception:
            logger.exception("Failed to load %s model", self._model_name)
            raise

    @staticmethod
    def _configure_intrinsics(
        intrinsics: Any,
        model_config: ModelConfig,
    ) -> Any:
        if model_config.task and not intrinsics.task:
            intrinsics.task = model_config.task
        if not intrinsics.labels and model_config.labels:
            label_file = files("picam") / "labels" / model_config.labels
            intrinsics.labels = label_file.read_text().splitlines()
        if model_config.bbox_normalization:
            intrinsics.bbox_normalization = model_config.bbox_normalization
        if model_config.bbox_order != "yx":
            intrinsics.bbox_order = model_config.bbox_order
        if model_config.ignore_dash_labels:
            intrinsics.ignore_dash_labels = model_config.ignore_dash_labels
        if model_config.softmax:
            intrinsics.softmax = model_config.softmax
        if intrinsics.ignore_dash_labels and intrinsics.labels:
            intrinsics.labels = [
                label for label in intrinsics.labels if label and label != "-"
            ]
        labels = intrinsics.labels
        imagenet_labels = files("picam") / "labels" / "imagenet_labels.txt"
        imagenet_count = len(imagenet_labels.read_text().splitlines())
        if (
            intrinsics.task == "classification"
            and labels
            and len(labels) == imagenet_count
        ):
            intrinsics.labels = labels[1:]
        return intrinsics

    def start(self, **config: Any) -> None:
        """Start the Picamera2 preview with optional configuration overrides."""
        logger.info("Starting camera...")
        try:
            from picamera2 import Picamera2

            if not hasattr(self, "_camera"):
                self._camera = Picamera2()

            picam_config = self._camera.create_preview_configuration(**config)
            self._camera.pre_callback = self._draw_results
            self._camera.post_callback = self._draw_overlay
            self._camera.start(picam_config, show_preview=False)
            logger.info("Started camera")
        except Exception:
            logger.exception("Failed to start camera")
            raise

    def upload_progress(self) -> Generator[float, None, None]:
        """Yield firmware upload progress as a fraction from 0.0 to 1.0.

        Yields:
            Upload progress fraction.

        """
        while True:
            current, total = self._imx500.get_fw_upload_progress(2)
            yield current / total if total else 0
            if total and current >= 0.95 * total:
                self.show_overlay(3)
                break
            time.sleep(0.5)

    def capture(self, stream: str = "main") -> tuple[dict, np.ndarray]:
        """Capture a single frame and return its metadata and pixel array.

        Returns:
            Tuple of metadata dict and frame array.

        """
        with self._camera.captured_request() as request:
            return request.get_metadata(), request.make_array(stream)

    def update(self, metadata: dict) -> None:
        """Parse inference outputs from metadata and update results."""
        self.results = self._parse_results(metadata)

    def show_overlay(self, duration: float = 0) -> None:
        """Show the overlay for a fixed duration, or toggle it if duration is 0."""
        if duration:
            self._show_overlay_until = time.time() + duration
        else:
            self._overlay_visible = not self._overlay_visible

    def stop(self) -> None:
        """Stop the camera and clear its callbacks."""
        logger.info("Stopping camera...")
        try:
            if not hasattr(self, "_camera"):
                return
            self._camera.stop()
            self._camera.pre_callback = None
            self._camera.post_callback = None
            logger.info("Stopped camera")
        except Exception:
            logger.exception("Failed to stop camera")
            raise

    def close(self) -> None:
        """Close and release the Picamera2 instance."""
        logger.info("Closing camera...")
        try:
            if not hasattr(self, "_camera"):
                return
            self._camera.close()
            logger.info("Closed camera")
        except Exception:
            logger.exception("Failed to close camera")
            raise

    def _draw_overlay(self, request: Any, stream: str = "main") -> None:
        if not self._overlay_visible and time.time() > self._show_overlay_until:
            return

        from picamera2 import MappedArray

        with MappedArray(request, stream) as m:
            if m.array is None:
                return
            draw_overlay(m.array, self._model_name, self.fps)

    def _draw_results(self, request: Any, stream: str = "main") -> None:
        from picamera2 import MappedArray

        with MappedArray(request, stream) as m:
            if m.array is None:
                return
            if self._intrinsics.task == "classification":
                draw_classifications(m.array, self.results)
            elif self._intrinsics.task == "object detection":
                draw_detections(m.array, self.results)
            elif self._intrinsics.task == "pose estimation":
                draw_poses(m.array, self.results)
            elif self._intrinsics.task == "segmentation":
                draw_masks(m.array, self.results)

    def _parse_results(self, metadata: dict) -> list:
        if not self._intrinsics.task:
            return []

        outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if not outputs:
            return self.results

        labels = self._intrinsics.labels or []

        if self._intrinsics.task == "classification":
            return parse_classifications(
                outputs,
                labels,
                softmax=bool(self._intrinsics.softmax),
            )

        if self._intrinsics.task == "object detection":
            return parse_detections(
                outputs,
                labels,
                self._display_size,
                self._input_size,
                bbox_normalization=bool(self._intrinsics.bbox_normalization),
                bbox_order=self._intrinsics.bbox_order or "yx",
            )

        if self._intrinsics.task == "pose estimation":
            return parse_poses(outputs, self._display_size)

        if self._intrinsics.task == "segmentation":
            return parse_masks(outputs, labels, self._display_size)

        return []


class Battery:
    """Reads charge state from the INA219 battery monitor over I2C."""

    BATTERY_ADDR = 0x43
    CURRENT_LSB = 0.1524
    POWER_LSB = 0.003048
    REG_BUSVOLTAGE = 0x02
    REG_CURRENT = 0x04
    REG_POWER = 0x03

    def __init__(self, address: int = BATTERY_ADDR, bus_number: int = 1) -> None:
        """Initialise the I2C bus connection to the battery monitor."""
        self._address = address
        self._bus = smbus2.SMBus(bus_number)

    def get_percent(self) -> int:
        """Return battery charge as a percentage from 0 to 100.

        Returns:
            Battery percentage.

        """
        voltage = self.get_voltage()
        percent = (voltage - 3.0) * 100 / 1.05  # calibrated for 1.05 V range
        return round(max(0, min(100, percent)))

    def get_voltage(self) -> float:
        """Return the bus voltage in volts.

        Returns:
            Bus voltage in volts.

        """
        return (self._read_register(self.REG_BUSVOLTAGE) >> 3) * 0.004

    def _read_register(self, register: int) -> int:
        data = self._bus.read_i2c_block_data(self._address, register, 2)
        return (data[0] * 256) + data[1]
