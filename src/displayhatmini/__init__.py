"""DisplayHAT Mini driver using lgpio.

Vendored and modified from https://github.com/pimoroni/displayhatmini-python
Original: Copyright (c) 2021 Pimoroni Ltd, MIT License
Changes: replaced RPi.GPIO with lgpio
"""

import contextlib
import types
from collections.abc import Callable

import lgpio
from st7789 import ST7789

__all__ = ["DisplayHATMini"]

_CHIP = 0  # 0 for Pi Zero


def _st7789_send(
    st7789: ST7789,
    data: object,
    is_data: bool = True,
    _chunk_size: object = None,
) -> None:
    # writebytes2 avoids a list copy on each SPI transfer, needed at 90 MHz
    st7789.set_pin(st7789._dc, is_data)
    if isinstance(data, int):
        data = [data & 0xFF]
    st7789._spi.writebytes2(data)


def _st7789_display(st7789: ST7789, image: object) -> None:
    # rot90 in the default display() adds ~10 ms per frame; skip it via MADCTL
    st7789.set_window()
    st7789.data(st7789.image_to_data(image, rotation=0))


class DisplayHATMini:
    """Hardware interface for the Pimoroni DisplayHAT Mini."""

    # User buttons
    BUTTON_A = 5
    BUTTON_B = 6
    BUTTON_X = 16
    BUTTON_Y = 24

    # Onboard RGB LED
    LED_R = 17
    LED_G = 27
    LED_B = 22

    # LCD Pins
    SPI_PORT = 0
    SPI_CS = 1
    SPI_DC = 9
    BACKLIGHT = 13

    # LCD Size
    WIDTH = 320
    HEIGHT = 240

    _LED_FREQ = 2000

    def __init__(self, buffer: object, *, backlight_pwm: bool = False) -> None:
        """Initialise the display, GPIO pins, LEDs, and ST7789 driver."""
        self.buffer = buffer
        self._h = lgpio.gpiochip_open(_CHIP)
        self._callbacks = []

        for pin in (self.BUTTON_A, self.BUTTON_B, self.BUTTON_X, self.BUTTON_Y):
            lgpio.gpio_claim_input(self._h, pin, lgpio.SET_PULL_UP)

        for pin in (self.LED_R, self.LED_G, self.LED_B):
            lgpio.gpio_claim_output(self._h, pin)
            lgpio.tx_pwm(self._h, pin, self._LED_FREQ, 100)

        if backlight_pwm:
            lgpio.gpio_claim_output(self._h, self.BACKLIGHT)
            lgpio.tx_pwm(self._h, self.BACKLIGHT, 500, 100)
            self._backlight_pwm = True
        else:
            self._backlight_pwm = False

        self.st7789 = ST7789(
            port=self.SPI_PORT,
            cs=self.SPI_CS,
            dc=self.SPI_DC,
            backlight=None if backlight_pwm else self.BACKLIGHT,
            width=self.WIDTH,
            height=self.HEIGHT,
            rotation=180,
            spi_speed_hz=90 * 1000 * 1000,
        )
        self.st7789.send = types.MethodType(_st7789_send, self.st7789)  # ty:ignore[invalid-assignment]
        self.st7789.command(0x36)
        self.st7789.data(0xB0)
        self.st7789.display = types.MethodType(_st7789_display, self.st7789)  # ty:ignore[invalid-assignment]

    def __del__(self) -> None:
        """Close the gpiochip handle on deletion."""
        with contextlib.suppress(Exception):
            lgpio.gpiochip_close(self._h)

    def set_led(self, r: float = 0, g: float = 0, b: float = 0) -> None:
        """Set the RGB LED colour with each channel from 0.0 to 1.0.

        Raises:
            ValueError: If any channel is outside the range 0.0 to 1.0.

        """
        if r < 0.0 or r > 1.0:
            msg = "r must be in the range 0.0 to 1.0"
            raise ValueError(msg)
        if g < 0.0 or g > 1.0:
            msg = "g must be in the range 0.0 to 1.0"
            raise ValueError(msg)
        if b < 0.0 or b > 1.0:
            msg = "b must be in the range 0.0 to 1.0"
            raise ValueError(msg)
        lgpio.tx_pwm(self._h, self.LED_R, self._LED_FREQ, (1.0 - r) * 100)
        lgpio.tx_pwm(self._h, self.LED_G, self._LED_FREQ, (1.0 - g) * 100)
        lgpio.tx_pwm(self._h, self.LED_B, self._LED_FREQ, (1.0 - b) * 100)

    def set_backlight(self, value: float) -> None:
        """Set the backlight brightness from 0.0 (off) to 1.0 (full)."""
        if self._backlight_pwm:
            lgpio.tx_pwm(self._h, self.BACKLIGHT, 500, value * 100)
        else:
            self.st7789.set_backlight(int(value))

    def on_button_pressed(self, callback: Callable[[int], None]) -> None:
        """Register a callback to fire when any button changes state."""
        for pin in (self.BUTTON_A, self.BUTTON_B, self.BUTTON_X, self.BUTTON_Y):
            lgpio.gpio_claim_alert(self._h, pin, lgpio.BOTH_EDGES, lgpio.SET_PULL_UP)
            cb = lgpio.callback(
                self._h,
                pin,
                lgpio.BOTH_EDGES,
                lambda _chip, gpio, _level, _tick: callback(gpio),
            )
            self._callbacks.append(cb)

    def read_button(self, pin: int) -> bool:
        """Return True if the button at the given pin is currently pressed.

        Returns:
            Button state.

        """
        return not lgpio.gpio_read(self._h, pin)

    def display(self) -> None:
        """Push the current buffer to the screen."""
        self.st7789.display(self.buffer)
