/*
  ESP32-CAM (AI-Thinker) -> 주기적으로 사진을 찍어 PC 파이썬 서버로 POST

  보드 설정: ESP32 Wrover Module (또는 AI Thinker ESP32-CAM), PSRAM Enabled
*/
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ───────── 사용자 설정 ─────────
const char* WIFI_SSID = "shy03";
const char* WIFI_PASS = "abc030902";
const char* SERVER_URL = "http://192.168.0.3:8000/upload";  // PC의 IP로 변경
const uint32_t CAPTURE_INTERVAL_MS = 1000;                   // 촬영 간격
// ───────────────────────────────

// AI-Thinker 핀맵
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;  // 항상 최신 프레임

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;  // 640x480 (사람 감지에 충분, 더 필요하면 SVGA/XGA)
    config.jpeg_quality = 12;           // 낮을수록 고화질
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("카메라 초기화 실패: 0x%x\n", err);
    return false;
  }
  return true;
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("WiFi 연결 중: %s ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);  // 지연 감소
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - start > 20000) {  // 20초 안에 안 되면 재시작
      Serial.println("\n연결 실패, 재시작");
      ESP.restart();
    }
  }
  Serial.printf("\n연결됨. IP: %s\n", WiFi.localIP().toString().c_str());
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!initCamera()) {
    delay(3000);
    ESP.restart();
  }
  connectWiFi();

  // 노출/화이트밸런스 안정화를 위해 첫 몇 장 버리기
  for (int i = 0; i < 5; i++) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(100);
  }
}

void loop() {
  uint32_t t0 = millis();
  connectWiFi();

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("촬영 실패");
    delay(500);
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(8000);
  int code = http.POST(fb->buf, fb->len);
  if (code > 0) {
    Serial.printf("[%d] %u bytes -> %s\n", code, (unsigned)fb->len, http.getString().c_str());
  } else {
    Serial.printf("전송 실패: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  esp_camera_fb_return(fb);

  uint32_t elapsed = millis() - t0;
  if (elapsed < CAPTURE_INTERVAL_MS) delay(CAPTURE_INTERVAL_MS - elapsed);
}
