// Driftlock Feather Demo — RPR (Reciprocal Ping Ranging) using RFM95 (LoRa)
// Dummy‑proof: flash SAME sketch to both boards. Role is chosen by a jumper.
//
// Hardware (either mix is fine):
//  - Adafruit Feather 32u4 LoRa (RFM95 onboard)
//  - Adafruit Feather M0 LoRa  (RFM95 onboard)
//  - Or Feather + RFM95 FeatherWing wired to default pins
//
// Role Selection:
//  - Leave A0 floating  => Initiator
//  - Tie A0 to GND      => Responder
//
// Arduino Libraries:
//  - Library Manager → install "RadioHead" by Mike McCauley
//
// Serial Output (initiator):
//   {"seq":N, "t_us":<float>, "rtt_us":<float>}

#include <Arduino.h>
#include <SPI.h>
#include <RH_RF95.h>

// Default pins per Adafruit boards
// 32u4 LoRa:   CS=8, IRQ=7, RST=4
// M0  LoRa:    CS=8, IRQ=3, RST=4

#define RFM95_CS   8
#define RFM95_RST  4

#if defined(ARDUINO_AVR_FEATHER32U4)
  #define RFM95_IRQ 7
#else
  // Covers M0, M4, RP2040, ESP32 variants commonly wired like the M0 LoRa
  #define RFM95_IRQ 3
#endif

RH_RF95 rf95(RFM95_CS, RFM95_IRQ);

// Demo config
static const float   RADIO_FREQ_MHZ   = 915.0;     // Set to 433/868/915 for your region
static const int16_t RADIO_TX_POWER   = 13;        // dBm (use 13–20; obey local regulations)
static const uint8_t ROLE_PIN         = A0;        // strap to GND for Responder
static const uint32_t PING_PERIOD_MS  = 20;        // ~50 pings/s
static const uint32_t RESP_TIMEOUT_MS = 50;        // up to 50 ms wait for PONG

// Packet layout
// PING: 'P' (1) + seq (4)
// PONG: 'O' (1) + seq (4) + t_rx_ping_us (4)

static void radioReset() {
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);
  digitalWrite(RFM95_RST, LOW);
  delay(10);
}

static bool radioInit() {
  radioReset();
  if (!rf95.init()) return false;
  // LoRa settings: 125 kHz BW, CR 4/5, SF 128 (7) — good default
  rf95.setModemConfig(RH_RF95::Bw125Cr45Sf128);
  rf95.setTxPower(RADIO_TX_POWER, true); // highPower=true uses PA_BOOST
  if (!rf95.setFrequency(RADIO_FREQ_MHZ)) return false;
  return true;
}

static uint32_t seq = 0;

void setup() {
  pinMode(ROLE_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  delay(1500);
  bool ok = radioInit();
  Serial.print(F("{\"radio\":\"RFM95\",\"ok\":"));
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
      if (rf95.available()) { uint8_t dump[8]; uint8_t len=sizeof(dump); rf95.recv(dump,&len); }
      return;
    }
    lastPingMs = nowMs;

    // Build and send PING
    uint8_t ping[1+4];
    ping[0] = 'P';
    memcpy(&ping[1], &seq, 4);
    const uint32_t t_tx_us = micros();
    rf95.send(ping, sizeof(ping));
    rf95.waitPacketSent();

    // Wait for PONG
    const uint32_t deadline = millis() + RESP_TIMEOUT_MS;
    bool got = false; uint8_t rx[1+4+4]; uint8_t n = sizeof(rx);
    uint32_t t_rx_us = 0; uint32_t t_rx_ping_us_remote = 0; uint32_t respSeq = 0;
    while ((int32_t)(deadline - millis()) > 0) {
      if (rf95.available()) {
        n = sizeof(rx); if (rf95.recv(rx, &n)) {
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
    // Responder: listen for PING, reply with PONG
    if (!rf95.available()) return;
    uint8_t rx[16]; uint8_t n = sizeof(rx);
    if (!rf95.recv(rx, &n)) return;
    if (n >= 1+4 && rx[0] == 'P') {
      uint32_t inSeq = 0; memcpy(&inSeq, &rx[1], 4);
      uint32_t t_rx_us = micros();
      uint8_t pong[1+4+4]; pong[0] = 'O';
      memcpy(&pong[1], &inSeq, 4);
      memcpy(&pong[1+4], &t_rx_us, 4);
      rf95.send(pong, sizeof(pong)); rf95.waitPacketSent();
    }
  }
}

