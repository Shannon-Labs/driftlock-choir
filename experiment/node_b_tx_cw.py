"""
Node B (CW attempt): Enable SX127x LoRa TX continuous mode at 915.001 MHz (Δf = 1 kHz).
Runs on Feather 32u4 with RFM95 (CircuitPython).

Warning: Low‑level register writes; benchtop only; keep power low.
"""
import time
import board
import busio
import digitalio
import adafruit_rfm9x


def enable_tx_continuous(rfm: adafruit_rfm9x.RFM9x):
    # Standby, LoRa
    reg_opmode = rfm._read_u8(0x01)
    rfm._write_u8(0x01, (reg_opmode | 0x80) & 0xF8 | 0x01)

    # Set frequency to 915.001 MHz
    rfm.frequency_mhz = 915.001
    rfm.tx_power = 5

    # Enable TxContinuousMode (RegModemConfig2 bit 3)
    m2 = rfm._read_u8(0x1E)
    m2 |= (1 << 3)
    rfm._write_u8(0x1E, m2)

    # Enter TX mode
    rfm._write_u8(0x01, (rfm._read_u8(0x01) & 0xF8) | 0x03)
    # Verify bits
    op = rfm._read_u8(0x01)
    m2v = rfm._read_u8(0x1E)
    tx_mode = op & 0x07
    tx_cont = (m2v >> 3) & 0x1
    print("RegOpMode=0x%02X mode=%d (3=TX), RegModemConfig2=0x%02X TX_CONT=%d" % (op, tx_mode, m2v, tx_cont))


def main():
    spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
    cs = digitalio.DigitalInOut(getattr(board, "RFM9X_CS", board.D8))
    reset = digitalio.DigitalInOut(getattr(board, "RFM9X_RST", board.D4))
    rfm = adafruit_rfm9x.RFM9x(spi, cs, reset, 915.001)
    print("Node B (CW): Enabling LoRa TX continuous mode @ 915.001 MHz")
    enable_tx_continuous(rfm)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
