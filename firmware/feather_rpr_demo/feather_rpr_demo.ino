// Driftlock Feather Demo — Reciprocal Ping Ranging (RPR) handshake
// Radio-agnostic sketch: fill in radioSend() / radioRecv() for your module
// Prints JSON over Serial: {"seq":N, "t_us": <float>, "rtt_us": <float>}

#include <Arduino.h>

// CONFIG
static const uint32_t PING_PERIOD_MS = 20; // adjust for your radio
static const uint32_t STARTUP_DELAY_MS = 2000;
static const bool IS_INITIATOR = true; // set one board false

// Radio stubs (replace with real implementations)
bool radioInit() { return true; }
bool radioSend(const uint8_t* data, size_t len) { (void)data; (void)len; return true; }
// Returns number of bytes received into buf (<= buflen), 0 if none
size_t radioRecv(uint8_t* buf, size_t buflen) { (void)buf; (void)buflen; return 0; }

// Packet formats (very simple)
// PING:  'P', seq (uint32_t)
// PONG:  'O', seq (uint32_t), t_rx_ping_us (uint32_t)

static uint32_t seq = 0;

void setup() {
  Serial.begin(115200);
  delay(STARTUP_DELAY_MS);
  if (!radioInit()) {
    Serial.println(F("{\"error\":\"radio_init_failed\"}"));
  }
  Serial.print(F("{\"role\":\""));
  Serial.print(IS_INITIATOR ? F("initiator") : F("responder"));
  Serial.println(F("\"}"));
}

void loop() {
  static uint32_t lastPingMs = 0;
  if (IS_INITIATOR) {
    const uint32_t nowMs = millis();
    if (nowMs - lastPingMs >= PING_PERIOD_MS) {
      lastPingMs = nowMs;
      // Send PING
      uint8_t pkt[1 + 4];
      pkt[0] = 'P';
      memcpy(&pkt[1], &seq, 4);
      const uint32_t t_tx_us = micros();
      radioSend(pkt, sizeof(pkt));

      // Wait for PONG with same seq (simple busy-wait)
      uint8_t rx[1 + 4 + 4];
      const uint32_t timeout_us = 50000; // 50 ms
      const uint32_t start_us = micros();
      bool got = false;
      uint32_t t_rx_us = 0;
      uint32_t t_rx_ping_us_remote = 0;
      while ((micros() - start_us) < timeout_us) {
        size_t n = radioRecv(rx, sizeof(rx));
        if (n >= 1 + 4 + 4 && rx[0] == 'O') {
          uint32_t s = 0; memcpy(&s, &rx[1], 4);
          if (s == seq) {
            memcpy(&t_rx_ping_us_remote, &rx[1+4], 4);
            t_rx_us = micros();
            got = true; break;
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
    }
  } else {
    // Responder: listen for PING, reply immediately with PONG
    uint8_t rx[1 + 4 + 4];
    size_t n = radioRecv(rx, sizeof(rx));
    if (n >= 1 + 4 && rx[0] == 'P') {
      uint32_t s = 0; memcpy(&s, &rx[1], 4);
      const uint32_t t_rx_us = micros();
      uint8_t tx[1 + 4 + 4];
      tx[0] = 'O';
      memcpy(&tx[1], &s, 4);
      memcpy(&tx[1+4], &t_rx_us, 4);
      radioSend(tx, sizeof(tx));
    }
  }
}

