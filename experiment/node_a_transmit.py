"""
Node A: Transmit at 915.000 MHz
Runs on Feather M0 with RFM95 LoRa (CircuitPython)

Notes:
- For some boards, replace board.RFM9X_CS / board.RFM9X_RST with explicit pins (e.g., board.D8 / board.D4).
- This example keeps the RF active by sending back‑to‑back packets. For true CW, enable SX127x TX CW test mode per datasheet.
"""
import time
import board
import busio
import digitalio
import adafruit_rfm9x

# SPI and control pins
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
cs = digitalio.DigitalInOut(getattr(board, "RFM9X_CS", board.D8))
reset = digitalio.DigitalInOut(getattr(board, "RFM9X_RST", board.D4))

# Initialize radio at 915.000 MHz
rfm9x = adafruit_rfm9x.RFM9x(spi, cs, reset, 915.0)
rfm9x.tx_power = 5  # low power for bench testing

print("Node A: Transmitting continuous tone-ish at 915.000 MHz")
print("Creates the reference frequency (LoRa TX active)")

# Keep RF chain active with back-to-back frames
payload = bytes([0xFF]) * 252
while True:
    rfm9x.send(payload)
    time.sleep(0.001)

