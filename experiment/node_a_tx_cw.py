"""
Node A (CW attempt): Enable SX127x LoRa TX continuous mode at 915.000 MHz.
Runs on Feather M0 with RFM95 (CircuitPython).

Warning: This uses low‑level register writes. Keep TX power low and operate in a shielded/benchtop environment.
References: Semtech SX1276/77/78/79 datasheet (LoRa TX continuous mode via RegModemConfig2.TX_CONT).
"""
import time
import board
import busio
import digitalio
import adafruit_rfm9x


def enable_tx_continuous(rfm: adafruit_rfm9x.RFM9x):
    # Go to standby
    reg_opmode = rfm._read_u8(0x01)
    # LongRangeMode bit7=1 (LoRa), Mode=STDBY (0x01 -> 0b10000001)
    rfm._write_u8(0x01, (reg_opmode | 0x80) & 0xF8 | 0x01)

    # Set frequency using library API for safety
    rfm.frequency_mhz = 915.000

    # Set TX power low
    rfm.tx_power = 5

    # Enable TxContinuousMode bit (RegModemConfig2, addr 0x1E, bit 3)
    m2 = rfm._read_u8(0x1E)
    m2 |= (1 << 3)
    rfm._write_u8(0x1E, m2)

    # Enter TX mode (RegOpMode Mode=TX)
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
    rfm = adafruit_rfm9x.RFM9x(spi, cs, reset, 915.0)

    print("Node A (CW): Enabling LoRa TX continuous mode @ 915.000 MHz")
    enable_tx_continuous(rfm)

    # Keep running to maintain TX
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
