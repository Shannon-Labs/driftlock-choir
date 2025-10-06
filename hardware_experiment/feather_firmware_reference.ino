/*
 * Adafruit Feather M4 Express - Reference Signal Generator
 * Part of E1 Chronometric Interferometry Hardware Experiment
 * 
 * This firmware generates a precise 1.000000 MHz reference signal
 * for the hardware implementation of the E1 beat note experiment.
 * 
 * Hardware: Adafruit Feather M4 Express
 * Signal Output: Pin A0 (DAC)
 * Frequency: 1.000000 MHz (base frequency)
 * Communication: USB Serial for timing coordination
 * 
 * The 433rd harmonic (433 MHz) will be captured by RTL-SDR
 */

#include <Arduino.h>

// Pin definitions
const int DAC_PIN = A0;           // DAC output pin
const int LED_PIN = LED_BUILTIN;  // Status LED

// Signal generation parameters
const float BASE_FREQ_HZ = 1000000.0;  // 1.000000 MHz base frequency
const uint32_t DAC_RESOLUTION = 4096;   // 12-bit DAC
const uint32_t SAMPLE_RATE = 120000000; // 120 MHz system clock

// Timing and control variables
volatile bool signal_active = false;
volatile uint32_t signal_start_time = 0;
volatile uint32_t sample_counter = 0;

// Waveform generation
const uint32_t WAVE_TABLE_SIZE = 1000;
uint16_t sine_table[WAVE_TABLE_SIZE];
uint32_t phase_accumulator = 0;
uint32_t phase_increment;

// Status tracking
unsigned long last_status_print = 0;
const unsigned long STATUS_INTERVAL = 1000; // 1 second

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial && millis() < 5000) {
    delay(10); // Wait for serial connection or timeout
  }
  
  Serial.println("=== Feather M4 Reference Signal Generator ===");
  Serial.println("E1 Chronometric Interferometry Hardware Experiment");
  Serial.println("Base Frequency: 1.000000 MHz");
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Initialize DAC
  analogWriteResolution(12); // 12-bit resolution
  analogWrite(DAC_PIN, DAC_RESOLUTION / 2); // Start at midpoint
  
  // Initialize sine wave table
  generate_sine_table();
  
  // Calculate phase increment for desired frequency
  // phase_increment = (desired_freq / sample_rate) * 2^32
  phase_increment = (uint32_t)((BASE_FREQ_HZ / SAMPLE_RATE) * 4294967296.0);
  
  // Setup timer for precise signal generation
  setup_timer();
  
  Serial.println("Initialization complete. Send 'START' to begin signal generation.");
  Serial.println("Commands: START, STOP, STATUS, FREQ");
}

void loop() {
  // Handle serial commands
  handle_serial_commands();
  
  // Print status periodically
  if (millis() - last_status_print > STATUS_INTERVAL) {
    print_status();
    last_status_print = millis();
  }
  
  // Blink LED to show activity
  if (signal_active) {
    digitalWrite(LED_PIN, (millis() / 100) % 2);
  } else {
    digitalWrite(LED_PIN, (millis() / 1000) % 2);
  }
  
  delay(10);
}

void generate_sine_table() {
  Serial.println("Generating sine wave table...");
  
  for (uint16_t i = 0; i < WAVE_TABLE_SIZE; i++) {
    float angle = (2.0 * PI * i) / WAVE_TABLE_SIZE;
    float sine_value = sin(angle);
    
    // Convert to DAC range (0 to DAC_RESOLUTION-1)
    // Add offset to center around DAC_RESOLUTION/2
    sine_table[i] = (uint16_t)((sine_value + 1.0) * (DAC_RESOLUTION / 2));
    
    // Ensure we don't exceed DAC range
    if (sine_table[i] >= DAC_RESOLUTION) {
      sine_table[i] = DAC_RESOLUTION - 1;
    }
  }
  
  Serial.print("Sine table generated with ");
  Serial.print(WAVE_TABLE_SIZE);
  Serial.println(" samples");
}

void setup_timer() {
  Serial.println("Setting up timer for signal generation...");
  
  // Use TC3 timer for precise timing
  // This will interrupt at sample rate to update DAC
  
  // Enable GCLK0 for TC3
  GCLK->PCHCTRL[TC3_GCLK_ID].reg = GCLK_PCHCTRL_GEN_GCLK0_Val | GCLK_PCHCTRL_CHEN;
  while (GCLK->PCHCTRL[TC3_GCLK_ID].bit.CHEN == 0);
  
  // Reset TC3
  TC3->COUNT16.CTRLA.bit.SWRST = 1;
  while (TC3->COUNT16.SYNCBUSY.bit.SWRST);
  
  // Configure TC3 for 16-bit mode
  TC3->COUNT16.CTRLA.reg = TC_CTRLA_MODE_COUNT16 | TC_CTRLA_PRESCALER_DIV1;
  
  // Set compare value for desired sample rate
  // 120MHz / desired_sample_rate - 1
  uint16_t compare_value = (uint16_t)(120000000 / (BASE_FREQ_HZ * WAVE_TABLE_SIZE)) - 1;
  TC3->COUNT16.CC[0].reg = compare_value;
  
  // Enable compare match interrupt
  TC3->COUNT16.INTENSET.bit.MC0 = 1;
  
  // Enable TC3 interrupt in NVIC
  NVIC_EnableIRQ(TC3_IRQn);
  
  Serial.print("Timer configured for ");
  Serial.print(BASE_FREQ_HZ * WAVE_TABLE_SIZE);
  Serial.println(" Hz sample rate");
}

void start_signal_generation() {
  Serial.println("Starting signal generation...");
  
  // Reset phase accumulator and counters
  phase_accumulator = 0;
  sample_counter = 0;
  signal_start_time = millis();
  
  // Enable signal generation
  signal_active = true;
  
  // Start timer
  TC3->COUNT16.CTRLA.bit.ENABLE = 1;
  while (TC3->COUNT16.SYNCBUSY.bit.ENABLE);
  
  Serial.println("Signal generation started!");
}

void stop_signal_generation() {
  Serial.println("Stopping signal generation...");
  
  // Disable timer
  TC3->COUNT16.CTRLA.bit.ENABLE = 0;
  while (TC3->COUNT16.SYNCBUSY.bit.ENABLE);
  
  // Set DAC to midpoint
  analogWrite(DAC_PIN, DAC_RESOLUTION / 2);
  
  signal_active = false;
  
  Serial.println("Signal generation stopped.");
  Serial.print("Total samples generated: ");
  Serial.println(sample_counter);
  Serial.print("Total duration: ");
  Serial.print(millis() - signal_start_time);
  Serial.println(" ms");
}

void handle_serial_commands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    if (command == "START") {
      if (!signal_active) {
        start_signal_generation();
      } else {
        Serial.println("Signal generation already active!");
      }
    }
    else if (command == "STOP") {
      if (signal_active) {
        stop_signal_generation();
      } else {
        Serial.println("Signal generation not active!");
      }
    }
    else if (command == "STATUS") {
      print_detailed_status();
    }
    else if (command == "FREQ") {
      Serial.print("Current frequency: ");
      Serial.print(BASE_FREQ_HZ / 1000000.0, 6);
      Serial.println(" MHz");
    }
    else if (command.startsWith("FREQ ")) {
      // Allow frequency adjustment (within limits)
      float new_freq = command.substring(5).toFloat() * 1000000.0;
      if (new_freq > 500000.0 && new_freq < 2000000.0) {
        update_frequency(new_freq);
      } else {
        Serial.println("Frequency must be between 0.5 and 2.0 MHz");
      }
    }
    else if (command == "HELP") {
      print_help();
    }
    else {
      Serial.println("Unknown command. Type HELP for available commands.");
    }
  }
}

void update_frequency(float new_freq_hz) {
  Serial.print("Updating frequency to ");
  Serial.print(new_freq_hz / 1000000.0, 6);
  Serial.println(" MHz");
  
  // Recalculate phase increment
  phase_increment = (uint32_t)((new_freq_hz / SAMPLE_RATE) * 4294967296.0);
  
  Serial.println("Frequency updated!");
}

void print_status() {
  if (signal_active) {
    Serial.print("ACTIVE - Runtime: ");
    Serial.print((millis() - signal_start_time) / 1000.0, 1);
    Serial.print("s, Samples: ");
    Serial.println(sample_counter);
  }
}

void print_detailed_status() {
  Serial.println("=== Reference Signal Generator Status ===");
  Serial.print("Signal State: ");
  Serial.println(signal_active ? "ACTIVE" : "STOPPED");
  Serial.print("Base Frequency: ");
  Serial.print(BASE_FREQ_HZ / 1000000.0, 6);
  Serial.println(" MHz");
  Serial.print("Target Harmonic: 433 MHz (");
  Serial.print(433000000.0 / BASE_FREQ_HZ, 0);
  Serial.println("th harmonic)");
  
  if (signal_active) {
    Serial.print("Runtime: ");
    Serial.print((millis() - signal_start_time) / 1000.0, 1);
    Serial.println(" seconds");
    Serial.print("Samples Generated: ");
    Serial.println(sample_counter);
    Serial.print("Average Sample Rate: ");
    Serial.print(sample_counter / ((millis() - signal_start_time) / 1000.0), 0);
    Serial.println(" Hz");
  }
  
  Serial.print("Free Memory: ");
  Serial.print(freeMemory());
  Serial.println(" bytes");
  Serial.println("=====================================");
}

void print_help() {
  Serial.println("=== Available Commands ===");
  Serial.println("START    - Start signal generation");
  Serial.println("STOP     - Stop signal generation");
  Serial.println("STATUS   - Show detailed status");
  Serial.println("FREQ     - Show current frequency");
  Serial.println("FREQ x.x - Set frequency to x.x MHz");
  Serial.println("HELP     - Show this help");
  Serial.println("=========================");
}

// Timer interrupt handler - generates the actual waveform
void TC3_Handler() {
  // Clear interrupt flag
  TC3->COUNT16.INTFLAG.bit.MC0 = 1;
  
  if (signal_active) {
    // Get current waveform sample
    uint32_t table_index = (phase_accumulator >> 22) % WAVE_TABLE_SIZE; // Use upper 10 bits
    uint16_t dac_value = sine_table[table_index];
    
    // Output to DAC
    analogWrite(DAC_PIN, dac_value);
    
    // Update phase accumulator
    phase_accumulator += phase_increment;
    sample_counter++;
  }
}

// Memory monitoring function
extern "C" char* sbrk(int incr);
int freeMemory() {
  char top;
  return &top - reinterpret_cast<char*>(sbrk(0));
}