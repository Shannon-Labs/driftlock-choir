# pico_firmware_reference.py
# Firmware for the Reference Raspberry Pi Pico

import time
import sys
import machine
from rp2 import PIO, StateMachine, asm_pio

# Define the PIO program for generating a square wave.
# This program toggles an output pin. The frequency is determined
# by the state machine's clock frequency and the instruction delays.
@asm_pio(set_init=PIO.OUT_LOW)
def square_wave():
    wrap_target()
    set(pins, 1) [15]  # Set pin high, wait 16 cycles
    set(pins, 0) [15]  # Set pin low, wait 16 cycles
    wrap()             # Loop forever

class SignalGenerator:
    """Controls a PIO state machine to generate a square wave on a pin."""

    def __init__(self, pin_num, freq_hz):
        self.pin = machine.Pin(pin_num)
        self.base_freq = freq_hz
        self.is_active = False

        # Total cycles in the PIO program is 32 (16 + 16).
        # The state machine frequency is set to freq_hz * 32 to achieve the target output frequency.
        self.sm = StateMachine(0, square_wave, freq=int(freq_hz * 32), set_base=self.pin)

    def start(self):
        if not self.is_active:
            self.sm.active(1)
            self.is_active = True
            print("Signal generation started!")

    def stop(self):
        if self.is_active:
            self.sm.active(0)
            self.pin.value(0)  # Ensure pin is left in a low state
            self.is_active = False
            print("Signal generation stopped.")

    def get_status(self):
        print("=== Reference Pico Status ===")
        print(f"Signal State: {'ACTIVE' if self.is_active else 'STOPPED'}")
        print(f"Base Frequency: {self.base_freq / 1e6:.6f} MHz")
        print(f"Target Harmonic: 433.0 MHz (433rd harmonic)")
        print("=============================")

def main():
    """Main loop to listen for serial commands."""
    # Use GPIO 15 for the signal output pin
    SIGNAL_PIN = 15
    # This is the reference Pico, generating the base frequency
    REFERENCE_FREQ = 1000000.0  # 1.0 MHz

    generator = SignalGenerator(SIGNAL_PIN, REFERENCE_FREQ)
    
    print("=== Pico Reference Signal Generator ===")
    print("Send START, STOP, or STATUS over serial.")

    # Use a simple polling mechanism for reading commands from stdin
    while True:
        command = sys.stdin.readline().strip().upper()
        if command == "START":
            generator.start()
        elif command == "STOP":
            generator.stop()
        elif command == "STATUS":
            generator.get_status()
        
        # Small delay to prevent busy-waiting
        time.sleep(0.05)

if __name__ == "__main__":
    main()
