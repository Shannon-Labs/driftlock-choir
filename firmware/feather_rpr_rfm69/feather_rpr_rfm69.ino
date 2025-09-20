// Driftlock Feather Demo — RPR (Reciprocal Ping Ranging) using RFM69 FeatherWing
// Dummy‑proof: flash SAME sketch to both boards. Role is chosen by a jumper.
//
// Hardware
// - Adafruit Feather (M0/M4/RP2040/ESP32‑S3 all fine)
// - Adafruit RFM69HCW FeatherWing (e.g., 900 MHz for US)
// - One short jumper wire:
//     • Leave pin A0 floating  => Initiator
//     • Tie pin A0 to GND      => Responder
//
// Library
// - Arduino IDE → Library Manager → install "RadioHead" by Mike McCauley
// - Select your Feather board in Tools, pick correct port, then Upload
//
// Serial Output
//   {"seq":N, "t_us":<float>, "rtt_us":<float>}
//
// Notes
// - Uses broadcast addressing — perfect for a 2‑board demo
// - Keep antennas attached even in short‑range demos; or cable + attenuators

#include <Arduino.h>
#include <SPI.h>
#include <RH_RF69.h>

// FeatherWing RFM69 default pins (Feather M0/M4/ESP32‑S3 compatible)
#if defined(ARDUINO_SAMD_FEATHER_M0) || defined(ADAFRUIT_FEATHER_M4_EXPRESS) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
  #define RFM69_CS   8
  #define RFM69_INT  3
  #define RFM69_RST  4
#else
  #define RFM69_CS   8
  #define RFM69_INT  3
  #define RFM69_RST  4
#endif

RH_RF69 rf69(RFM69_CS, RFM69_INT);

// Demo config
static const float   RADIO_FREQ_MHZ   = 915.0;     // set 433/868/915 to match your wing
static const int16_t RADIO_TX_POWER   = 14;        // dBm (up to 20 with PA boost)
static const uint8_t ROLE_PIN         = A0;        // strap to GND for Responder
static const uint32_t PING_PERIOD_MS  = 20;        // ~50 pings/s target
static const uint32_t RESP_TIMEOUT_MS = 50;        // wait up to 50ms for PONG

// Packet layout
// PING: 'P' (1) + seq (4)
// PONG: 'O' (1) + seq (4) + t_rx_ping_us (4)

static uint32_t seq = 0;

static void radioReset() {
  pinMode(RFM69_RST, OUTPUT);
  digitalWrite(RFM69_RST, LOW);
  delay(10);
  digitalWrite(RFM69_RST, HIGH);
  delay(10);
  digitalWrite(RFM69_RST, LOW);
  delay(10);
}

static bool radioInit() {
  radioReset();
  if (!rf69.init()) return false;
  rf69.setTxPower(RADIO_TX_POWER, true);  // highPower=true uses PA boost on HCW
  // RadioHead API uses Hz; setFrequency takes MHz
  if (!rf69.setFrequency(RADIO_FREQ_MHZ)) return false;
  // Optional: set modem config for robustness
  // rf69.setModemConfig(RH_RF69::GFSK_Rb125Fd125);
  return true;
}

void setup() {
  pinMode(ROLE_PIN, INPUT_PULLUP); // HIGH=floating (initiator), LOW=GND (responder)
  Serial.begin(115200);
  delay(1500);
  bool ok = radioInit();
  Serial.print(F("{\"radio\":\"RFM69\",\"ok\":"));
  Serial.print(ok ? F("true") : F("false"));
  Serial.print(F(",\"freq_mhz\":"));
  Serial.print(RADIO_FREQ_MHZ, 1);
  Serial.print(F(",\"role\":\""));
  Serial.print(digitalRead(ROLE_PIN) == LOW ? F("responder") : F("initiator"));
  Serial.println(F("\"}"));
}

void loop() {
  const bool isResponder = (digitalRead(ROLE_PIN) == LOW);

  if (!isResponder) {
    // Initiator
    static uint32_t lastPingMs = 0;
    const uint32_t nowMs = millis();
    if (nowMs - lastPingMs < PING_PERIOD_MS) {
      // Also check for stray packets to keep FIFO clear
      if (rf69.available()) { uint8_t buf[32]; uint8_t len=sizeof(buf); rf69.recv(buf,&len); }
      return;
    }
    lastPingMs = nowMs;

    // Build and send PING
    uint8_t ping[1+4];
    ping[0] = 'P';
    memcpy(&ping[1], &seq, 4);
    const uint32_t t_tx_us = micros();
    rf69.send(ping, sizeof(ping));
    rf69.waitPacketSent();

    // Wait for PONG
    const uint32_t deadline = millis() + RESP_TIMEOUT_MS;
    bool got = false; uint8_t rx[1+4+4]; uint8_t n = sizeof(rx);
    uint32_t t_rx_us = 0; uint32_t t_rx_ping_us_remote = 0; uint32_t respSeq = 0;
    while ((int32_t)(deadline - millis()) > 0) {
      if (rf69.available()) {
        n = sizeof(rx); if (rf69.recv(rx, &n)) {
          if (n >= 1+4+4 && rx[0] == 'O') {
            memcpy(&respSeq, &rx[1], 4);
            if (respSeq == seq) {
              memcpy(&t_rx_ping_us_remote, &rx[1+4], 4);
              t_rx_us = micros(); got = true; break;
            }
          }
        }
      }
      yield();
    }

    if (got) {
      const float rtt_us = float(t_rx_us - t_tx_us);
      const float t_s = 0.001f * float(nowMs);
      Serial.print(F("{\"seq\":")); Serial.print(seq);
      Serial.print(F(",\"t_us\":")); Serial.print(t_s * 1e6f, 1);
      Serial.print(F(",\"rtt_us\":")); Serial.print(rtt_us, 1);
      Serial.println(F("}"));
    }
    seq++;
  } else {
    // Responder: listen for PING, reply immediately with PONG
    if (!rf69.available()) return;
    uint8_t rx[16]; uint8_t n = sizeof(rx);
    if (!rf69.recv(rx, &n)) return;
    if (n >= 1+4 && rx[0] == 'P') {
      uint32_t inSeq = 0; memcpy(&inSeq, &rx[1], 4);
      uint32_t t_rx_us = micros();
      uint8_t pong[1+4+4]; pong[0] = 'O';
      memcpy(&pong[1], &inSeq, 4);
      memcpy(&pong[1+4], &t_rx_us, 4);
      rf69.send(pong, sizeof(pong)); rf69.waitPacketSent();
    }
  }
}

