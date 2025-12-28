# 개발 로그 (Development Log)

> **목적**: AI 개발 도우미와의 대화가 중단되더라도 개발 연속성을 유지하기 위한 로그 파일
>
> **사용법**: 새로운 대화 시작 시 이 파일을 참조하여 이전 개발 내용을 이어서 진행

---

## 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Production Dual NPU Application |
| **개발사** | MetaVu Co., Ltd. |
| **개발자** | JINSONG ROH |
| **이메일** | enjoydays@metavu.io |
| **홈페이지** | www.metavu.io |
| **개발 시작일** | 2025년 12월 26일 |
| **현재 버전** | 1.0.0 |

---

## 핵심 파일 구조

```
/home/orangepi/dual_npu_demo/
├── production_app.py          # 메인 애플리케이션 (148KB, 3700+ lines)
├── rk3588_vlm_server.py       # RK3588 로컬 VLM API 서버
├── README.md                  # 프로젝트 문서
├── DEVLOG.md                  # 개발 로그 (이 파일)
├── .dev.vars                  # API 키 설정 파일
├── .gitignore                 # Git 제외 파일 설정
├── run_production.sh          # 실행 스크립트
├── scripts/
│   ├── backup.sh              # 타임스탬프 백업 스크립트
│   ├── rollback.sh            # 버전 롤백 스크립트
│   └── version_manager.sh     # 통합 버전 관리 도구
└── backups/                   # 백업 폴더 (버전별 백업)
```

---

## 개발 이력

### [2025-12-28] v1.1.0 - RK3588 로컬 VLM 지원

#### 완료된 기능

8. **RK3588 로컬 VLM 모델 지원**
   - 파일: `rk3588_vlm_server.py` (신규), `production_app.py`
   - 설명: RK3588 NPU에서 로컬로 실행되는 Vision-Language Model 지원
   - 지원 모델 (9종):
     - Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen3-VL-2B
     - MiniCPM-V-2.6
     - InternVL2-1B, InternVL3-1B
     - Janus-Pro-1B, SmolVLM, DeepSeek-OCR
   - 기능:
     - FastAPI 기반 OpenAI 호환 API 서버
     - RKLLM/RKLLAMA 런타임 통합
     - API 키 없이 로컬에서 무료 실행
   - 사용법:
     ```bash
     # VLM 서버 시작
     python3 rk3588_vlm_server.py --port 8088

     # 또는 RKLLAMA 서버
     rkllama_server --models ~/rkllm_models
     ```

---

### [2025-12-28] v1.0.0 - 초기 릴리스

#### 완료된 기능

1. **다중 인물 인식 색상 구분**
   - 파일: `production_app.py`
   - 위치: `PERSON_COLORS` 변수 및 `draw_boxes()` 함수
   - 설명: 최대 10명까지 각기 다른 색상으로 바운딩 박스 표시
   - 색상: Green, Blue, Red, Cyan, Magenta, Yellow, Orange, Purple, Spring Green, Light Coral

2. **다국어 UI 지원 (10개 언어)**
   - 파일: `production_app.py`
   - 위치: `UI_LANGUAGES`, `UI_TRANSLATIONS`, `get_text()` 함수
   - 지원 언어: 한국어, English, 日本語, 中文, Español, Français, Deutsch, Português, Русский, العربية
   - 설정 다이얼로그에서 언어 변경 가능

3. **빠른 버튼 프롬프트 다국어 지원**
   - 파일: `production_app.py`
   - 위치: `quick_btn_data` 및 버튼 클릭 핸들러
   - 설명: "뭐가 보여?", "분석해줘", "설명해줘" 버튼이 선택된 UI 언어로 프롬프트 입력

4. **LLM 응답 언어 동기화**
   - 파일: `production_app.py`
   - 위치: `LLMWorker.run()` 메서드의 vision_prompt 구성부
   - 설명: UI 언어 설정에 따라 LLM이 해당 언어로 응답

5. **채팅 내용 유지 (언어 변경 시)**
   - 파일: `production_app.py`
   - 위치: `self.chat_has_messages` 플래그, `apply_translations()` 메서드
   - 설명: UI 언어 변경 시 기존 대화 내용 보존

6. **기본 LLM 모델 설정**
   - 파일: `production_app.py`
   - 위치: `LLMWorker.__init__()`, `model_combo.setCurrentText()`
   - 기본값: Claude Sonnet

7. **상세 주석 추가**
   - 전체 코드에 한국어 주석 추가
   - 파일 헤더에 개발자 정보 포함

---

## 현재 아키텍처

### 주요 클래스

| 클래스명 | 역할 |
|----------|------|
| `VisionLLMClient` | Vision LLM API 통합 클라이언트 (Gemini, Claude, GPT-4o, Groq) |
| `DXM1Detector` | DeepX DX-M1 NPU 기반 객체 감지 |
| `LLMWorker` | QThread 기반 LLM 요청 처리 워커 |
| `ProductionApp` | 메인 PyQt5 애플리케이션 (QMainWindow) |

### 주요 기능 흐름

```
카메라 입력 → DXM1Detector(NPU 추론) → 객체 감지 결과
                                          ↓
                                    draw_boxes() → 화면 표시
                                          ↓
                        사용자 질문 → LLMWorker → VisionLLMClient → LLM API
                                          ↓
                                    응답 표시 + TTS 재생
```

### API 키 설정 (.dev.vars)

```bash
GEMINI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
```

---

## 알려진 이슈

| 이슈 | 상태 | 비고 |
|------|------|------|
| libGL error (rockchip driver) | 무시 가능 | 기능에 영향 없음 |
| ALSA underrun warning | 무시 가능 | 오디오 버퍼 관련, 간헐적 발생 |
| OpenAI stream deprecation warning | 무시 가능 | 기능 정상 동작 |

---

## 다음 개발 계획 (TODO)

- [ ] 객체 인식 모델 추가 (YOLOv9 등)
- [ ] 제스처 인식 기능
- [ ] 얼굴 인식 및 감정 분석
- [ ] 음성 명령 인식 개선
- [ ] 다중 카메라 지원

---

## 개발 환경

| 항목 | 버전/사양 |
|------|-----------|
| 하드웨어 | Orange Pi 5B (RK3588 SoC) |
| NPU | DeepX DX-M1 (USB 연결) |
| OS | Linux 6.1.43-rockchip-rk3588 |
| Python | 3.x |
| PyQt5 | 5.x |
| 카메라 | USB 웹캠 (/dev/video0) |

---

## 버전 관리 시스템

### 버전 관리 도구 사용법

```bash
# 통합 버전 관리 도구 실행 (메뉴 방식)
./scripts/version_manager.sh

# 빠른 백업 생성
./scripts/backup.sh

# 이전 버전으로 롤백
./scripts/rollback.sh
```

### 버전 관리 도구 기능

| 기능 | 설명 |
|------|------|
| 백업 생성 | 타임스탬프 기반 백업 파일 생성 |
| 백업 목록 | 저장된 모든 백업 파일 확인 |
| 버전 롤백 | 선택한 백업 버전으로 복원 |
| 버전 비교 | 현재 파일과 백업 파일 diff 비교 |
| Git 커밋 | 현재 변경사항 Git 커밋 |
| Git 로그 | 커밋 이력 확인 |
| Git 롤백 | 특정 Git 커밋으로 복원 |
| 오래된 백업 정리 | 30일 이상 된 백업 자동 삭제 |

### Git 커밋 이력

| 커밋 | 날짜 | 설명 |
|------|------|------|
| 21c39c7 | 2025-12-28 | RK3588 로컬 VLM 모델 지원 추가 |
| 83ee4a1 | 2025-12-28 | 개발 지침서 및 보안 시스템 추가 |
| b7d555d | 2025-12-28 | 버전 관리 시스템 추가 |
| d65c512 | 2025-12-28 | v1.0.0: Initial commit |

### 백업 파일

| 파일명 | 날짜 | 설명 |
|--------|------|------|
| `production_app_v20251228_201147.py` | 2025-12-28 20:11 | RK3588 로컬 VLM 모델 지원 추가 |
| `production_app_v20251228_194309.py` | 2025-12-28 19:43 | 자동설명 다국어 지원 추가 |
| `production_app_v20251228_190036.py` | 2025-12-28 19:00 | 버전 관리 시스템 추가 후 |
| `production_app_backup_20251228_183819.py` | 2025-12-28 18:38 | 주석 추가 전 백업 |

### 긴급 복구 방법

```bash
# 방법 1: 백업 파일에서 복원
cp backups/production_app_v{원하는버전}.py production_app.py

# 방법 2: Git에서 복원
git checkout {커밋해시} -- production_app.py

# 방법 3: 롤백 스크립트 사용
./scripts/rollback.sh
```

---

## 새 대화 시작 시 참고사항

1. **이 파일(DEVLOG.md)을 먼저 읽어주세요**
2. **production_app.py의 파일 헤더 주석 확인**
3. **README.md에서 전체 기능 개요 확인**
4. **git log로 최근 변경사항 확인**

### 빠른 컨텍스트 복원 명령

```bash
# 프로젝트 구조 확인
ls -la /home/orangepi/dual_npu_demo/

# 최근 Git 커밋 확인
cd /home/orangepi/dual_npu_demo && git log --oneline -10

# 개발 로그 확인
cat /home/orangepi/dual_npu_demo/DEVLOG.md

# 애플리케이션 실행
cd /home/orangepi/dual_npu_demo && ./run_production.sh
```

---

## 로그 업데이트 기록

| 날짜 | 작성자 | 내용 |
|------|--------|------|
| 2025-12-28 | Claude AI | 초기 DEVLOG.md 생성 |
| 2025-12-28 | Claude AI | 버전 관리 시스템 추가 (Git + 백업 스크립트) |
| 2025-12-28 | Claude AI | 개발 지침서(DEV_GUIDELINES.md) 생성 |
| 2025-12-28 | Claude AI | 보안 시스템 추가 (pre-commit hook, security_check.sh) |
| 2025-12-28 | Claude AI | 자동 설명 프롬프트 다국어 지원 추가 |
| 2025-12-28 | Claude AI | GitHub 저장소 생성 및 푸시 (JinsongRoh/dual_npu_demo) |

---

## 새 AI 대화 시작 시 복사할 프롬프트

```
이전 개발 대화가 중단되어 새로 시작합니다.

프로젝트 위치: /home/orangepi/dual_npu_demo/

먼저 다음 파일들을 읽어서 이전 개발 내용을 파악해주세요:
1. DEVLOG.md - 개발 로그 (필수)
2. README.md - 프로젝트 개요
3. git log --oneline -10 - 최근 변경사항

이전 개발 내용을 파악한 후, [여기에 요청사항 작성] 작업을 진행해주세요.
```

---

> **참고**: 이 로그 파일은 개발 진행 시마다 업데이트하여 개발 연속성을 유지하세요.
>
> **중요**: 주요 기능 개발 완료 후 반드시 `./scripts/backup.sh` 실행하여 백업하세요.
