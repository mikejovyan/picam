"""Main application for Raspberry Pi AI camera with Display HAT Mini."""

import logging
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import tomllib

from .hardware import Battery, Camera, Display, ModelConfig

logger = logging.getLogger(__name__)


class Picam:
    """Top-level application coordinating the display, camera, and battery."""

    CONFIG_FILE = Path.home() / "picam" / "config.toml"
    DISPLAY_SIZE = (320, 240)  # (DisplayHATMini.WIDTH, DisplayHATMini.HEIGHT)
    PICTURES_DIR = Path.home() / "picam" / "pictures"
    RAW_SIZE = (2028, 1520)  # sensor size: (4056, 3040)

    def __init__(
        self,
        display: Display,
        camera: Camera,
        battery: Battery,
    ) -> None:
        """Initialise Picam with hardware interfaces and model list."""
        self.display = display
        self.camera = camera
        self.battery = battery
        self.pictures_dir = self.PICTURES_DIR

        input_tensor_only = [ModelConfig(name="inputtensoronly")]
        try:
            with self.CONFIG_FILE.open("rb") as f:
                data = tomllib.load(f)
            self.models = [
                ModelConfig(**m) for m in data.get("models", [])
            ] or input_tensor_only
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            self.models = input_tensor_only

        self.running: bool = False
        self._frame_count: int = 0
        self._last_battery_check: float = 0.0
        self._last_fps_time: float = 0.0
        self._model_index: int = 0

        # systemctl stop sends SIGTERM, not SIGINT
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def run(self, stream: str = "main") -> None:
        """Start the main loop: load model, capture frames, and handle input."""
        logger.info("Starting Picam...")
        model_config = self.models[self._model_index]

        try:
            self.display.start()
            self.display.on_button_pressed(self._handle_button)
            self.display.message("Loading...")
            self.camera.load_model(model_config)
            self.camera.start(**self._camera_config())
            for progress in self.camera.upload_progress():
                self.display.message(f"Loading {model_config.name}...", progress)
            self.running = True
            self._last_fps_time = time.time()
            show_thread: threading.Thread | None = None

            while self.running:
                if not self.camera.is_switching:
                    metadata, frame = self.camera.capture(stream)
                    self.camera.update(metadata)
                    if show_thread is not None:
                        show_thread.join()
                    self.display.set_buffer(frame)
                    self._update_fps()
                    self._update_battery_led()
                    show_thread = threading.Thread(
                        target=self.display.show,
                        daemon=True,
                    )
                    show_thread.start()
        finally:
            logger.info("Cleaning up...")
            self.display.stop()
            self.camera.stop()
            self.camera.close()
            logger.info("Cleaned up")

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self.running = False

    @staticmethod
    def _camera_config(
        main_size: tuple[int, int] = DISPLAY_SIZE,
        raw_size: tuple[int, int] = RAW_SIZE,
        main_format: str = "BGR888",
        raw_format: str = "SRGGB10_CSI2P",
        buffer_count: int = 12,
    ) -> dict:
        return {
            "main": {"size": main_size, "format": main_format},
            "raw": {"size": raw_size, "format": raw_format},
            "buffer_count": buffer_count,
        }

    def _capture_image(self) -> None:
        from .draw import save_image

        self.pictures_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        filename = self.pictures_dir / f"{timestamp}.jpg"

        logger.info("Capturing image...")
        try:
            save_image(self.display.get_buffer(), str(filename))
            logger.info("Captured image: %s", filename)
        except Exception:
            logger.exception("Failed to capture image")

    def _handle_button(self, pin: int) -> None:
        if not self.display.read_button(pin):
            return

        if pin == Display.BUTTON_Y:
            self.stop()
        elif pin == Display.BUTTON_B:
            self.camera.show_overlay()
        elif pin == Display.BUTTON_X:
            self._capture_image()
        elif pin == Display.BUTTON_A:
            self._switch_model()

    def _handle_sigterm(self, _sig: int, _frame: object) -> None:
        self.stop()

    def _switch_model(self) -> None:
        if len(self.models) <= 1:
            logger.warning("Cannot switch: only one model configured")
            return

        self._model_index = (self._model_index + 1) % len(self.models)
        model_config = self.models[self._model_index]

        logger.info("Switching model...")
        try:
            self.camera.is_switching = True
            time.sleep(0.1)  # wait for active captured_request to complete
            self.camera.stop()
            self.camera.results = []
            self.camera.load_model(model_config)
            self.camera.start(**self._camera_config())
            for progress in self.camera.upload_progress():
                self.display.message(f"Loading {model_config.name}...", progress)
            logger.info("Switched model")
        except Exception:
            logger.exception("Failed to switch model")
        finally:
            self.camera.is_switching = False

    def _update_battery_led(
        self,
        interval: float = 30,
        low_threshold: int = 20,
    ) -> None:
        now = time.time()
        if now - self._last_battery_check < interval:
            return
        self._last_battery_check = now

        percent = self.battery.get_percent()
        if percent > low_threshold:
            self.display.set_led(0.05, 0.05, 0.05)
        else:
            self.display.set_led(0.1, 0, 0)

    def _update_fps(self) -> None:
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self.camera.fps = int(self._frame_count / (now - self._last_fps_time))
            self._frame_count = 0
            self._last_fps_time = now


def main() -> None:
    """Initialise hardware, create Picam, and run until interrupted."""
    logging.basicConfig(level=logging.INFO)
    display = Display()
    camera = Camera()
    battery = Battery()
    picam = Picam(display=display, camera=camera, battery=battery)

    try:
        picam.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        picam.stop()


if __name__ == "__main__":
    main()
