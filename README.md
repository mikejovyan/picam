# picam

[![license](https://img.shields.io/github/license/mikejovyan/picam)](LICENSE)

Raspberry Pi AI camera for edge computer vision, with a small display and battery.

## Hardware

- [Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) with headers
- [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/)
- [Pimoroni Display HAT Mini](https://shop.pimoroni.com/products/display-hat-mini)
- [Waveshare UPS HAT (C)](https://www.waveshare.com/ups-hat-c.htm)

## Software

Follow the [official docs](https://www.raspberrypi.com/documentation/computers/getting-started.html#installing-the-operating-system) to install **Raspberry Pi OS (Legacy, 64-bit) Lite** using Raspberry Pi Imager. Configure your Wi-Fi credentials and enable SSH before flashing.

### System setup

Update the system:

```shell
sudo apt update
sudo apt full-upgrade
```

Enable SPI for the display:

```shell
sudo raspi-config nonint do_spi 0
```

Reboot:

```shell
sudo systemctl reboot
```

### Dependencies

Install the AI camera firmware and model files:

```shell
sudo apt install imx500-firmware imx500-models
```

Install the Python camera library without GUI dependencies:

```shell
sudo apt install python3-picamera2 --no-install-recommends
```

Install the I2C library for the battery monitor:

```shell
sudo apt install python3-smbus2
```

Install the GPIO library for the display and buttons:

```shell
sudo apt install python3-lgpio
```

Install git:

```shell
sudo apt install git
```

### Installation

Clone the repo:

```shell
git clone https://github.com/mikejovyan/picam.git ~/picam
```

Create a virtual environment:

```shell
python -m venv --system-site-packages ~/picam/.venv
```

Install the picam package:

```shell
~/picam/.venv/bin/pip install ~/picam
```

Install simplejpeg 1.9.0 or later, which is required for compatibility with numpy 2.x:

```shell
~/picam/.venv/bin/pip install "simplejpeg>=1.9.0"
```

Enable linger so the user systemd service starts at boot without an active login session:

```shell
loginctl enable-linger
```

Create the systemd service `~/.config/systemd/user/picam.service`:

```ini
[Unit]
Description=picam
After=default.target

[Service]
ExecStart=%h/picam/.venv/bin/picam
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
```

Enable the service:

```shell
systemctl --user daemon-reload && systemctl --user enable picam
```

Reboot:

```shell
sudo systemctl reboot
```

## Controls

| Button | Action |
|--------|--------|
| A | Switch AI model |
| B | Show/hide model info overlay |
| X | Capture image (saved to `~/picam/pictures/`) |
| Y | Exit |
