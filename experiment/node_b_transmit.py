"""
Node B: Transmit at 915.001 MHz (Δf = 1 kHz)
Runs on Feather 32u4 with RFM95 LoRa (CircuitPython)

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

# Intentional +1 kHz offset from Node A
rfm9x = adafruit_rfm9x.RFM9x(spi, cs, reset, 915.001)
rfm9x.tx_power = 5

print("Node B: Transmitting at 915.001 MHz (Δf = 1 kHz)")
print("The offset creates a ~1 kHz beat with Node A")

payload = bytes([0xFF]) * 252
while True:
    rfm9x.send(payload)
    time.sleep(0.001)

