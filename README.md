# Production Dual NPU Demo

듀얼 NPU를 활용한 실시간 객체 검출 및 AI 비전 어시스턴트 데모 애플리케이션

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![License](https://img.shields.io/badge/License-Proprietary-red.svg)

---

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [설치 방법](#설치-방법)
- [환경 설정](#환경-설정)
- [실행 방법](#실행-방법)
- [사용법](#사용법)
- [아키텍처](#아키텍처)
- [지원 언어](#지원-언어)
- [API 프로바이더](#api-프로바이더)
- [파일 구조](#파일-구조)
- [개발 정보](#개발-정보)

---

## 개요

이 프로그램은 두 개의 NPU(Neural Processing Unit)를 동시에 활용하는 데모 애플리케이션입니다:

### DX-M1 NPU (DeepX)
- **YOLOX-S 모델**을 사용한 실시간 객체 검출
- 최대 **400+ FPS** 추론 성능
- 최대 **10명**까지 다중 인식 지원
- 각 객체별 고유 색상 바운딩 박스
- 정적 이미지(사진) 자동 필터링

### RK3588 NPU (Rockchip)
- **Vision LLM** 채팅 기능
- 다양한 클라우드 LLM 프로바이더 지원
- **STT/TTS** 음성 입출력 지원
- 실시간 통역 모드

---

## 주요 기능

### 1. 실시간 객체 검출
- YOLOX-S 기반 고속 객체 검출
- 80개 COCO 클래스 지원 (사람, 자동차, 동물 등)
- IoU 기반 객체 추적 (ProIOUTracker)
- Weighted Box Fusion으로 정확도 향상

### 2. AI 비전 어시스턴트
- 카메라 영상 기반 질의응답
- 다중 LLM 프로바이더 지원
  - Google Gemini (무료/유료)
  - OpenAI GPT-4o
  - Anthropic Claude
  - Groq (고속 추론)
  - xAI Grok

### 3. 음성 기능
- **STT (Speech-to-Text)**: OpenAI Whisper API
- **TTS (Text-to-Speech)**: OpenAI TTS API
- 다국어 음성 인식 지원

### 4. 실시간 통역 모드
- 50+ 언어 간 실시간 통역
- 양방향 자동 언어 감지
- 음성 입출력 연동

### 5. 자동 장면 설명
- 설정한 간격으로 자동 장면 분석
- 음성 출력 옵션
- 5초 ~ 5분 간격 설정 가능

### 6. UI 다국어 지원
- 10개 언어 UI 지원
- 실시간 언어 전환
- 퀵 버튼 프롬프트 자동 번역

### 7. 시스템 모니터링
- CPU/GPU 사용률
- 메모리 사용량
- NPU 온도 및 클럭
- 전력 소비 추정

---

## 시스템 요구사항

### 하드웨어
| 구성요소 | 사양 |
|---------|------|
| 보드 | Orange Pi 5B (RK3588) |
| NPU 가속기 | DeepX DX-M1 |
| 메모리 | 8GB+ 권장 |
| 카메라 | USB 웹캠 또는 CSI 카메라 |
| 디스플레이 | 1920x1080 권장 |

### 소프트웨어
| 구성요소 | 버전 |
|---------|------|
| OS | Ubuntu 22.04 (aarch64) |
| Python | 3.8+ |
| PyQt5 | 5.15+ |
| OpenCV | 4.5+ |
| DeepX SDK | dx_rt 최신 |

### Python 패키지
```bash
pip3 install PyQt5 opencv-python numpy requests sounddevice soundfile openai
```

---

## 설치 방법

### 1. 저장소 클론
```bash
cd ~
git clone <repository_url> dual_npu_demo
cd dual_npu_demo
```

### 2. DeepX SDK 설치
```bash
# DeepX SDK가 설치되어 있어야 합니다
# 경로: /home/orangepi/deepx_sdk/dx_rt/python_package/src
```

### 3. 모델 파일 준비
```bash
# YOLOX-S 모델 파일을 다음 경로에 배치
# /home/orangepi/model_for_demo/YOLOXS-1.dxnn
```

### 4. 의존성 설치
```bash
pip3 install -r requirements.txt
```

---

## 환경 설정

### API 키 설정

`.dev.vars` 파일을 생성하여 API 키를 설정합니다:

```bash
# 파일 생성
nano ~/.dev.vars

# 또는 프로젝트 디렉토리에
nano /home/orangepi/dual_npu_demo/.dev.vars
```

### .dev.vars 파일 형식
```ini
# Google Gemini API (무료 티어 사용 가능)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API (GPT-4o, Whisper, TTS)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude API
ANTHROPIC_API_KEY=your_claude_api_key_here

# Groq API (선택)
GROQ_API_KEY=your_groq_api_key_here

# xAI Grok API (선택)
XAI_API_KEY=your_xai_api_key_here
```

### 보안 설정
```bash
# 파일 권한 설정 (소유자만 읽기/쓰기)
chmod 600 .dev.vars
```

---

## 실행 방법

### 기본 실행
```bash
cd /home/orangepi/dual_npu_demo
python3 production_app.py
```

### SSH 원격 실행
```bash
DISPLAY=:0.0 python3 production_app.py
```

### 백그라운드 실행
```bash
nohup python3 production_app.py &
```

---

## 사용법

### 화면 구성

```
+------------------------------------------+------------------------+
|                                          |                        |
|          비디오 패널 (왼쪽)               |     채팅 패널 (우상단)   |
|                                          |                        |
|    - 실시간 카메라 영상                   |    - AI 대화 내역       |
|    - 객체 검출 바운딩 박스                |    - 텍스트 입력        |
|    - FPS/추론시간/객체수                  |    - 음성 입력 버튼      |
|                                          |    - 퀵 버튼            |
|                                          |                        |
+------------------------------------------+------------------------+
|                                          |                        |
|                                          |   시스템 모니터 (우하단) |
|                                          |                        |
|                                          |    - CPU/GPU 사용률     |
|                                          |    - NPU 상태           |
|                                          |    - 온도/메모리        |
+------------------------------------------+------------------------+
```

### 버튼 기능

| 버튼 | 기능 |
|------|------|
| 🌐 통역 | 실시간 통역 모드 설정 |
| ⚙️ 설정 | UI 언어 및 자동 설명 설정 |
| 🎤 | 음성 입력 (길게 누르기) |
| 🔊 | 음성 출력 토글 |
| ➤ | 메시지 전송 |
| ⏱️ 자동 | 자동 장면 설명 토글 |

### 퀵 버튼
- **뭐가 보여?** - 현재 화면 설명 요청
- **분석해줘** - 상세 분석 요청
- **설명해줘** - 장면 설명 요청

### 단축키
| 단축키 | 기능 |
|--------|------|
| Enter | 메시지 전송 |
| 마우스 클릭 (🎤 버튼) | 음성 녹음 시작/종료 |

---

## 아키텍처

### 클래스 구조

```
production_app.py
│
├── VisionLLMClient          # Vision LLM API 클라이언트
│   ├── call_gemini()        # Google Gemini API
│   ├── call_openai()        # OpenAI GPT-4o API
│   ├── call_claude()        # Anthropic Claude API
│   ├── call_groq()          # Groq LLaVA API
│   └── call_xai()           # xAI Grok API
│
├── DXM1Detector             # DX-M1 NPU 객체 검출기
│   ├── initialize()         # NPU 초기화
│   ├── detect()             # 객체 검출 수행
│   ├── draw_boxes()         # 바운딩 박스 그리기
│   └── ProIOUTracker        # 객체 추적기
│
├── LLMWorker (QThread)      # LLM 처리 워커 스레드
│   ├── run()                # 메인 루프
│   ├── add_query()          # 쿼리 추가
│   └── update_frame()       # 프레임 업데이트
│
├── SystemMonitor (QThread)  # 시스템 모니터 워커
│   └── run()                # 시스템 상태 수집
│
├── SpeechToText (QThread)   # STT 워커
│   └── run()                # 음성 인식 처리
│
├── TextToSpeech (QThread)   # TTS 워커
│   └── run()                # 음성 합성 처리
│
├── AutoDescriptionWorker    # 자동 설명 워커
│   └── run()                # 주기적 장면 설명
│
└── ProductionApp (QMainWindow)  # 메인 애플리케이션
    ├── init_ui()            # UI 초기화
    ├── init_camera()        # 카메라 초기화
    ├── init_detector()      # 검출기 초기화
    ├── init_workers()       # 워커 스레드 시작
    ├── update_frame()       # 프레임 업데이트
    └── apply_translations() # 다국어 적용
```

### 데이터 흐름

```
카메라 → 프레임 캡처 → DX-M1 NPU 추론 → 객체 검출 → 화면 표시
                              ↓
                        검출 결과 컨텍스트
                              ↓
사용자 입력 → LLMWorker → Vision LLM API → 응답 → 채팅창 표시
                              ↓
                         TTS (선택)
                              ↓
                         음성 출력
```

---

## 지원 언어

### UI 언어 (10개)
| 코드 | 언어 |
|------|------|
| ko | 한국어 |
| en | English |
| ja | 日本語 |
| zh | 中文 |
| es | Español |
| fr | Français |
| de | Deutsch |
| pt | Português |
| ru | Русский |
| ar | العربية |

### 통역 지원 언어 (50+)
한국어, 영어, 일본어, 중국어(간체/번체), 스페인어, 프랑스어, 독일어,
이탈리아어, 포르투갈어, 러시아어, 아랍어, 힌디어, 베트남어, 태국어,
인도네시아어, 터키어, 폴란드어, 네덜란드어, 스웨덴어 등

---

## API 프로바이더

### Vision LLM 모델

| 모델명 | 프로바이더 | 모델 ID | 비용 |
|--------|-----------|---------|------|
| Gemini Flash | Google | gemini-2.0-flash | 무료 |
| Gemini Pro | Google | gemini-1.5-pro | 유료 |
| GPT-4o Vision | OpenAI | gpt-4o | 유료 |
| GPT-4o Mini | OpenAI | gpt-4o-mini | 저렴 |
| Claude Sonnet | Anthropic | claude-sonnet-4 | 유료 |
| Claude Haiku | Anthropic | claude-3-haiku | 저렴 |

### 기본 모델
- **Claude Sonnet** (claude-sonnet-4-20250514)

---

## 파일 구조

```
dual_npu_demo/
├── production_app.py           # 메인 애플리케이션 (3600+ lines)
├── production_app_backup_*.py  # 백업 파일
├── .dev.vars                   # API 키 설정 (숨김 파일)
├── README.md                   # 이 문서
└── requirements.txt            # Python 의존성
```

### 외부 의존성 경로
```
/home/orangepi/
├── deepx_sdk/
│   └── dx_rt/
│       └── python_package/
│           └── src/            # DeepX Python SDK
└── model_for_demo/
    ├── YOLOXS-1.dxnn           # YOLOX-S 모델 (권장)
    ├── YOLOv5s_640.dxnn        # YOLOv5s 모델
    └── YOLOv9-S-2.dxnn         # YOLOv9-S 모델
```

---

## 문제 해결

### 일반적인 문제

#### 1. Qt 플러그인 오류
```bash
# 환경 변수 설정
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
```

#### 2. 카메라 인식 안됨
```bash
# 카메라 장치 확인
ls -la /dev/video*

# 권한 설정
sudo chmod 666 /dev/video0
```

#### 3. NPU 초기화 실패
```bash
# DX-M1 상태 확인
dxrt-cli -s

# NPU 드라이버 확인
lsmod | grep deepx
```

#### 4. API 키 오류
```bash
# .dev.vars 파일 확인
cat ~/.dev.vars

# 권한 확인
ls -la ~/.dev.vars
```

### 로그 확인
프로그램 실행 시 터미널에 다음 로그가 출력됩니다:
- `[Config]` - 설정 로드 관련
- `[DX-M1]` - NPU 초기화/추론 관련
- `[LLM]` - LLM API 호출 관련
- `[STT]` - 음성 인식 관련
- `[TTS]` - 음성 출력 관련
- `[UI]` - UI 업데이트 관련

---

## 성능 최적화

### 권장 설정
- **카메라 해상도**: 640x480 또는 1280x720
- **프레임 레이트**: 30 FPS
- **JPEG 품질**: 85% (API 전송용)
- **추론 임계값**: 0.55 (기본값)

### NPU 최적화
- YOLOX-S 모델 사용 권장 (Apache 2.0 라이선스)
- 비-PPU 모델로 소프트웨어 NMS 수행
- 최대 10명 동시 인식으로 제한

---

## 개발 정보

| 항목 | 내용 |
|------|------|
| **회사명** | MetaVu Co., Ltd. |
| **개발자** | JINSONG ROH |
| **이메일** | enjoydays@metavu.io |
| **홈페이지** | www.metavu.io |
| **저작권** | © 2025 MetaVu Co., Ltd. |
| **최종수정** | 2025년 12월 28일 |

---

## 라이선스

이 소프트웨어는 MetaVu Co., Ltd.의 독점 소프트웨어입니다.
무단 복제, 배포, 수정을 금지합니다.

© 2025 MetaVu Co., Ltd. All rights reserved.

---

## 변경 이력

### v1.0.0 (2025-12-28)
- 초기 릴리스
- 듀얼 NPU 지원 (DX-M1 + RK3588)
- 다중 Vision LLM 프로바이더 지원
- UI 다국어 지원 (10개 언어)
- STT/TTS 음성 기능
- 실시간 통역 모드
- 자동 장면 설명 기능
- 다중 인식 지원 (최대 10명)
