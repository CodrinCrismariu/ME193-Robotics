#include <Wire.h>
#include <Arduino_RouterBridge.h>

// Step counter, MCU side. Samples the MPU6050 on the ankle at a steady 100 Hz
// and streams the raw accelerometer and gyro to Python, where a neural network
// counts steps and estimates distance. Nothing is filtered or fused here: the
// network was trained on a simulation of exactly this sensor, settings and
// filter (step-counter-training/simulate.py), so the board must send what the
// sensor saw.
//
// Sampling runs on its own thread: each Bridge.notify blocks for the few ms
// the 115200-baud link needs to send it, which would otherwise make the sample
// spacing uneven. Samples wait in a ring buffer and loop() sends them in
// batches of BATCH (20 notifies/s instead of 100).
//
// Wire format: Bridge.notify("imu_batch", seq, hex) where seq numbers the
// first sample and hex is BATCH samples of 6 little-endian int16:
//   ax ay az in mg, gx gy gz in 0.1 deg/s
// Fixed units keep the ranges below in this file only.

// ---------------------------------------------------------------------------
// IMU (bus/address probing as in the MPU6050 Testing app)
// ---------------------------------------------------------------------------
#define HAVE_WIRE1 1

static TwoWire *imuBus = &Wire;
static uint8_t MPU_ADDR = 0x68;
static const char *busName = "?";
static uint8_t whoAmI = 0;

static const uint8_t REG_SMPLRT_DIV   = 0x19;
static const uint8_t REG_CONFIG       = 0x1A;
static const uint8_t REG_GYRO_CONFIG  = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t REG_PWR_MGMT_1   = 0x6B;
static const uint8_t REG_WHO_AM_I     = 0x75;

// +/-16 g and +/-2000 deg/s, the widest ranges: at the ankle, running swings
// the shank past 1000 deg/s and footstrikes reach 10 g and more. The
// simulator clips at the same limits.
static const float ACCEL_LSB_PER_G  = 2048.0f;
static const float GYRO_LSB_PER_DPS = 16.4f;

static const uint32_t SAMPLE_INTERVAL_US = 10000;   // 100 Hz
static const int BATCH = 5;                         // samples per notify
static const int RING = 128;                        // 1.28 s of slack for a stalled link
static const unsigned long STATUS_INTERVAL_MS = 5000;

#define IMU_THREAD_PRIORITY  -1     // cooperative: runs as soon as its sleep ends
#define IMU_THREAD_STACK     2048
K_THREAD_STACK_DEFINE(imuStack, IMU_THREAD_STACK);
static struct k_thread imuThread;

struct Sample {
  uint32_t seq;
  int16_t v[6];
};
static Sample ring[RING];
static uint32_t head = 0;            // samples written (sampling thread)
static uint32_t tail = 0;            // samples sent (loop)
static struct k_mutex ringMutex;

static bool imuReady = false;
static const char *lastError = "";

// Diagnostics (sampling thread counts, loop() reports).
static volatile uint32_t samplesTaken = 0;
static volatile uint32_t samplesDropped = 0;
static volatile uint32_t maxGapUs = 0;

static bool writeReg(uint8_t reg, uint8_t value) {
  imuBus->beginTransmission(MPU_ADDR);
  imuBus->write(reg);
  imuBus->write(value);
  return imuBus->endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t *buf, uint8_t len) {
  imuBus->beginTransmission(MPU_ADDR);
  imuBus->write(reg);
  if (imuBus->endTransmission(false) != 0) {
    return false;
  }
  if (imuBus->requestFrom(MPU_ADDR, (size_t)len) != (size_t)len) {
    return false;
  }
  for (uint8_t i = 0; i < len; i++) {
    buf[i] = imuBus->read();
  }
  return true;
}

// Log to the App Lab Serial Monitor and mirror into the Python log.
// Never use Serial here: that UART belongs to the Router Bridge.
static void report(const char *msg) {
  Monitor.println(msg);
  Bridge.notify("imu_status", msg);
}

// Pick whichever bus answers at 0x68 or 0x69.
static bool findIMU() {
  struct { TwoWire *bus; const char *name; } candidates[] = {
    { &Wire, "Wire (SDA/SCL pins)" },
#if HAVE_WIRE1
    { &Wire1, "Wire1 (Qwiic)" },
#endif
  };
  for (auto &c : candidates) {
    for (uint8_t addr = 0x68; addr <= 0x69; addr++) {
      c.bus->beginTransmission(addr);
      if (c.bus->endTransmission() == 0) {
        imuBus = c.bus;
        busName = c.name;
        MPU_ADDR = addr;
        return true;
      }
    }
  }
  lastError = "MPU6050: not found on any I2C bus at 0x68/0x69";
  return false;
}

static bool initIMU() {
  if (!readRegs(REG_WHO_AM_I, &whoAmI, 1)) {
    lastError = "MPU6050: not found on any I2C bus at 0x68/0x69";
    return false;
  }
  writeReg(REG_PWR_MGMT_1, 0x80);    // device reset
  k_msleep(100);
  if (!writeReg(REG_PWR_MGMT_1, 0x01)) {   // wake up, clock from gyro X
    lastError = "MPU6050: found on I2C but did not accept configuration";
    return false;
  }
  k_msleep(10);
  writeReg(REG_SMPLRT_DIV, 0x00);    // 1 kHz internal rate
  writeReg(REG_CONFIG, 0x03);        // DLPF ~44 Hz: just under the 50 Hz Nyquist limit of 100 Hz sampling
  writeReg(REG_GYRO_CONFIG, 0x18);   // +/-2000 deg/s
  writeReg(REG_ACCEL_CONFIG, 0x18);  // +/-16 g
  k_msleep(50);
  return true;
}

static int16_t clamp16(float v) {
  return (int16_t)lroundf(constrain(v, -32767.0f, 32767.0f));
}

// One burst read of ACCEL_XOUT..GYRO_ZOUT, converted to the wire units.
static bool readSample(int16_t out[6]) {
  uint8_t raw[14];
  if (!readRegs(REG_ACCEL_XOUT_H, raw, sizeof(raw))) {
    return false;
  }
  for (int i = 0; i < 3; i++) {
    int16_t a = (int16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);
    int16_t g = (int16_t)((raw[8 + 2 * i] << 8) | raw[9 + 2 * i]);
    out[i] = clamp16(a * (1000.0f / ACCEL_LSB_PER_G));       // mg
    out[3 + i] = clamp16(g * (10.0f / GYRO_LSB_PER_DPS));    // 0.1 deg/s
  }
  return true;
}

static void imuEntry(void *, void *, void *) {
  uint32_t seq = 0;
  uint32_t last = micros();
  uint32_t next = last;
  while (true) {
    if (!imuReady) {
      imuReady = findIMU() && initIMU();
      if (!imuReady) {
        report(lastError);
        k_msleep(2000);     // keep retrying so plugging the sensor in later works
        continue;
      }
      last = next = micros();
    }

    Sample s;
    uint32_t now = micros();
    if (!readSample(s.v)) {
      report("MPU6050: read failed - sensor disconnected?");
      imuReady = false;
      continue;
    }
    uint32_t gap = now - last;
    last = now;
    if (gap > maxGapUs) maxGapUs = gap;
    s.seq = seq++;

    k_mutex_lock(&ringMutex, K_FOREVER);
    if (head - tail < RING) {
      ring[head % RING] = s;
      head++;
    } else {
      samplesDropped++;     // link stalled for over a second; Python sees the seq gap
    }
    k_mutex_unlock(&ringMutex);
    samplesTaken++;

    next += SAMPLE_INTERVAL_US;
    int32_t wait = (int32_t)(next - micros());
    if (wait > 0) {
      k_usleep(wait);
    } else {
      next = micros();     // fell behind: don't try to catch up
      k_yield();
    }
  }
}

// ---------------------------------------------------------------------------

static unsigned long lastStatusMs = 0;
static uint32_t batchesSent = 0;

void setup() {
  Bridge.begin();
  Monitor.begin();

  Wire.begin();
  Wire.setClock(400000);
#if HAVE_WIRE1
  Wire1.begin();
  Wire1.setClock(400000);
#endif

  k_mutex_init(&ringMutex);
  k_thread_create(&imuThread, imuStack, K_THREAD_STACK_SIZEOF(imuStack), imuEntry,
                  NULL, NULL, NULL, IMU_THREAD_PRIORITY, 0, K_NO_WAIT);
  k_thread_name_set(&imuThread, "imu");
  lastStatusMs = millis();
}

void loop() {
  // Send one batch once enough samples are waiting. Only consecutive samples
  // go in a batch, so Python can trust that seq..seq+n-1 are evenly spaced.
  Sample batch[BATCH];
  int n = 0;
  k_mutex_lock(&ringMutex, K_FOREVER);
  if (head - tail >= (uint32_t)BATCH) {
    uint32_t first = ring[tail % RING].seq;
    while (n < BATCH && ring[(tail + n) % RING].seq == first + n) {
      batch[n] = ring[(tail + n) % RING];
      n++;
    }
    tail += n;
  }
  k_mutex_unlock(&ringMutex);

  if (n > 0) {
    static const char HEX_DIGITS[] = "0123456789abcdef";
    char hex[BATCH * 12 * 2 + 1];
    int k = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < 6; j++) {
        uint16_t u = (uint16_t)batch[i].v[j];
        uint8_t bytes[2] = { (uint8_t)(u & 0xFF), (uint8_t)(u >> 8) };   // little-endian
        for (uint8_t b : bytes) {
          hex[k++] = HEX_DIGITS[b >> 4];
          hex[k++] = HEX_DIGITS[b & 0x0F];
        }
      }
    }
    hex[k] = '\0';
    Bridge.notify("imu_batch", batch[0].seq, (const char *)hex);
    batchesSent++;
  } else {
    k_msleep(2);
  }

  // Status every few seconds, so it reaches Python and the web page even
  // though anything printed at boot is sent before Python is listening.
  unsigned long nowMs = millis();
  if (nowMs - lastStatusMs >= STATUS_INTERVAL_MS) {
    float secs = (nowMs - lastStatusMs) / 1000.0f;
    String msg;
    if (imuReady) {
      msg = "MPU6050 on ";
      msg += busName;
      msg += " at 0x";
      msg += String(MPU_ADDR, HEX);
      msg += ", WHO_AM_I 0x";
      msg += String(whoAmI, HEX);
      msg += " | ";
      msg += String(samplesTaken / secs, 1);
      msg += " Hz, ";
      msg += String(batchesSent / secs, 1);
      msg += " batches/s, max gap ";
      msg += String(maxGapUs / 1000.0f, 1);
      msg += " ms, dropped ";
      msg += String(samplesDropped);
      report(msg.c_str());
    }
    samplesTaken = 0;
    batchesSent = 0;
    maxGapUs = 0;
    lastStatusMs = nowMs;
  }
}
