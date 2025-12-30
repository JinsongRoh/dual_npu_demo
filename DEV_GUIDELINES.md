# AI 기반 개발 작업 지침서 (Development Guidelines)

> **버전**: 1.0.0
> **작성일**: 2025년 12월 28일
> **회사**: MetaVu Co., Ltd.
> **작성자**: JINSONG ROH (Claude AI 협업)

---

## 목차

0. [프로젝트 개발 규칙 (필독)](#0-프로젝트-개발-규칙-필독)

1. [개요](#1-개요)
2. [프로젝트 컨텍스트 설정](#2-프로젝트-컨텍스트-설정)
3. [효과적인 프롬프트 작성법](#3-효과적인-프롬프트-작성법)
4. [개발 워크플로우](#4-개발-워크플로우)
5. [코드 품질 관리](#5-코드-품질-관리)
6. [버전 관리 및 백업](#6-버전-관리-및-백업)
7. [보안 지침](#7-보안-지침)
8. [문제 해결 가이드](#8-문제-해결-가이드)
9. [대화 연속성 유지](#9-대화-연속성-유지)
10. [참고 자료](#10-참고-자료)

---

## 0. 프로젝트 개발 규칙 (필독)

> **중요**: 이 섹션은 프로젝트 개발 시 반드시 준수해야 할 규칙입니다.

### 0.1 코드 작성 규칙

#### 주석 및 문서화
```
[필수] 모든 코드에 상세한 한국어 주석 작성
[필수] 파일 헤더에 개발자 정보 포함
[필수] 함수/클래스마다 용도 설명 주석

# 파일 헤더 예시
"""
================================================================================
파일명: example.py
설명: 이 파일의 기능 설명
================================================================================
회사명: MetaVu Co., Ltd.
개발자: JINSONG ROH
이메일: enjoydays@metavu.io
홈페이지: www.metavu.io
저작권: © 2025 MetaVu Co., Ltd. All rights reserved.
최종수정: YYYY년 MM월 DD일
================================================================================
"""
```

#### 민감 정보 관리 (절대 규칙)
```
[절대금지] API 키, 비밀번호 등 민감 정보 하드코딩 금지
[필수] 환경 변수 또는 .dev.vars 파일 사용
[필수] .gitignore에 민감 파일 등록 확인
[필수] 커밋 전 보안 검사 (./scripts/security_check.sh)
```

#### 다국어 지원
```
[필수] UI 텍스트는 번역 시스템(get_text()) 사용
[필수] 새 UI 요소 추가 시 UI_TRANSLATIONS에 모든 언어 번역 추가
[필수] LLM 응답도 UI 언어에 맞춰 출력되도록 프롬프트 작성
[필수] 빠른 버튼 등 사용자 입력도 선택된 언어로 처리
```

### 0.2 버전 관리 규칙

#### 백업 규칙
```
[필수] 주요 기능 완료 후 반드시 백업: ./scripts/backup.sh
[필수] 위험한 변경 전 백업 생성
[필수] 하루 작업 종료 시 Git 커밋 + 백업
[권장] 대화 중단 가능성 시 즉시 백업
```

#### Git 커밋 규칙
```
[필수] 커밋 메시지는 한글로 작성
[필수] 의미 있는 단위로 커밋 (기능별)
[필수] 커밋 전 보안 검사 통과 확인
[금지] 민감 정보 포함 파일 커밋 금지
```

#### 롤백 대비
```
[필수] 롤백 가능하도록 정기적 백업 유지
[권장] 주요 변경 전 현재 상태 태그 생성
[필수] 롤백 방법 숙지: ./scripts/rollback.sh
```

### 0.3 AI 개발 연속성 규칙

#### 대화 중단 대비
```
[필수] 주요 작업 완료 시 DEVLOG.md 업데이트
[필수] 미완료 작업 TODO로 명시
[필수] 다음 작업 계획 기록
[필수] 백업 생성
```

#### 새 대화 시작 시
```
[필수] DEVLOG.md 먼저 읽기 요청
[필수] README.md로 프로젝트 파악
[필수] git log로 최근 변경사항 확인
[권장] DEV_GUIDELINES.md 참조 요청
```

### 0.4 UI/UX 개발 규칙

#### 다중 객체 표시
```
[필수] 여러 객체 인식 시 각각 다른 색상 적용
[권장] 최대 10개 색상 팔레트 사용
[필수] ID 라벨과 바운딩 박스 색상 일치
```

#### 언어 변경 처리
```
[필수] UI 언어 변경 시 채팅 내용 유지
[필수] 플레이스홀더만 번역, 실제 메시지는 보존
[필수] 언어 변경 상태 플래그로 관리
```

#### 기본값 설정
```
[참고] LLM 모델 기본값: Claude Sonnet
[참고] UI 언어 기본값: 한국어 (ko)
[권장] 사용자 설정은 로컬 저장하여 유지
```

### 0.5 문서화 규칙

#### 필수 문서 파일
```
README.md       - 프로젝트 개요, 설치 방법, 사용법
DEVLOG.md       - 개발 로그, 대화 연속성 유지
DEV_GUIDELINES.md - 개발 지침서 (이 문서)
CLAUDE.md       - AI 작업 규칙 (선택)
```

#### 문서 업데이트 시점
```
[필수] 새 기능 추가 시 README.md 업데이트
[필수] 개발 작업 완료 시 DEVLOG.md 업데이트
[권장] 새 규칙 정립 시 DEV_GUIDELINES.md 업데이트
```

### 0.6 테스트 및 검증 규칙

```
[필수] 기능 구현 후 실행 테스트
[필수] UI 변경 후 스크린샷 확인
[권장] 다국어 변경 후 모든 언어에서 테스트
[필수] 에러 발생 시 로그 확인 및 수정
```

### 0.7 AI 모델 라이선스 규칙

> **중요**: 상업적 사용을 위해 반드시 라이선스를 확인하세요.

#### 허용 라이선스 (상업적 사용 가능)
```
[허용] Apache 2.0 - 상업적 사용 완전 허용
[허용] MIT - 상업적 사용 완전 허용
[허용] BSD-3-Clause - 상업적 사용 허용
[허용] CC0 / Public Domain - 제한 없음
```

#### 금지 라이선스 (상업적 사용 제한)
```
[금지] AGPL-3.0 - 소스 공개 의무, 상업적 사용 제한
[금지] GPL-3.0 - 파생작 소스 공개 의무
[금지] CC BY-NC - 비상업적 용도만 허용
[주의] LGPL - 동적 링크 시에만 허용, 정적 링크 시 소스 공개 필요
```

#### 모델별 라이선스 현황
```
# DX-M1 NPU 모델
YOLOX-S (객체 감지)         - Apache 2.0  ✅ 사용 가능
YOLOv5Pose (포즈 추정)      - AGPL-3.0   ❌ 사용 금지
YOLOv5/v7/v8              - AGPL-3.0   ❌ 사용 금지

# 권장 대안 모델
YOLOX-Pose                 - Apache 2.0  ✅ 권장 (DX-M1 변환 필요)
RTMPose (MMPose)           - Apache 2.0  ✅ 권장 (DX-M1 변환 필요)
MediaPipe Pose             - Apache 2.0  ✅ 권장

# RK3588 NPU 모델
Qwen2.5-VL-3B-RKLLM        - Apache 2.0  ✅ 사용 가능
```

#### 모델 사용 전 체크리스트
```
□ 모델 라이선스 확인 (GitHub/HuggingFace)
□ Apache 2.0, MIT, BSD 중 하나인지 확인
□ AGPL, GPL, CC BY-NC 라이선스는 사용 금지
□ 불확실한 경우 사용하지 않음
□ 사용 모델 목록을 문서에 기록
```

### 0.8 빠른 참조 명령어

```bash
# 백업 생성
./scripts/backup.sh

# 버전 롤백
./scripts/rollback.sh

# 버전 관리 (메뉴)
./scripts/version_manager.sh

# 보안 검사
./scripts/security_check.sh

# 앱 실행
./run_production.sh

# Git 상태 확인
git status && git log --oneline -5
```

---

## 1. 개요

### 1.1 문서 목적

이 문서는 AI 코딩 어시스턴트(Claude Code)를 활용한 개발 작업 시 최적의 결과를 얻기 위한 지침을 제공합니다. 프롬프트 엔지니어링, 워크플로우 최적화, 품질 관리 등 AI 기반 개발의 모범 사례를 포함합니다.

### 1.2 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **명확성** | 모호하지 않은 구체적인 지시 사항 제공 |
| **단계적 접근** | 복잡한 작업은 작은 단위로 분할 |
| **검증 우선** | AI 생성 코드는 반드시 검토 후 적용 |
| **컨텍스트 유지** | 충분한 배경 정보 제공 |
| **반복적 개선** | 피드백을 통한 지속적 품질 향상 |

---

## 2. 프로젝트 컨텍스트 설정

### 2.1 CLAUDE.md 파일 활용

Claude Code는 프로젝트 루트의 `CLAUDE.md` 파일을 자동으로 컨텍스트에 로드합니다.

```markdown
# CLAUDE.md 예시

## 프로젝트 개요
- 프로젝트명: Dual NPU Demo Application
- 언어: Python 3.x
- 프레임워크: PyQt5
- 하드웨어: Orange Pi 5B + DeepX DX-M1 NPU

## 코딩 스타일
- 들여쓰기: 4 spaces
- 문자열: 작은따옴표 우선 사용
- 주석: 한국어로 작성
- 함수명: snake_case
- 클래스명: PascalCase

## 금지 사항
- 하드코딩된 API 키 절대 금지
- print() 대신 logging 모듈 사용
- 전역 변수 최소화

## 테스트 명령
- pytest tests/
- python -m unittest discover

## 빌드/실행
- ./run_production.sh
```

### 2.2 하위 폴더 오버라이드

특정 폴더에 다른 규칙을 적용하려면 해당 폴더에 별도의 `CLAUDE.md`를 생성합니다.

```
project/
├── CLAUDE.md              # 전역 설정
├── src/
│   └── CLAUDE.md          # src 전용 설정 (오버라이드)
└── tests/
    └── CLAUDE.md          # tests 전용 설정
```

### 2.3 필수 컨텍스트 파일

| 파일 | 용도 |
|------|------|
| `DEVLOG.md` | 개발 이력 및 대화 연속성 |
| `README.md` | 프로젝트 개요 및 사용법 |
| `CLAUDE.md` | AI 작업 규칙 및 설정 |
| `DEV_GUIDELINES.md` | 개발 지침서 (이 문서) |

---

## 3. 효과적인 프롬프트 작성법

### 3.1 프롬프트 구조 (CIER 패턴)

```
[Context] 배경 정보 및 현재 상황
[Identity] AI의 역할 정의
[Examples] 원하는 결과의 예시
[Request] 구체적인 요청 사항
```

### 3.2 좋은 프롬프트 vs 나쁜 프롬프트

#### 나쁜 예시 (모호함)
```
버튼 추가해줘
```

#### 좋은 예시 (구체적)
```
production_app.py의 채팅 입력창 옆에 "초기화" 버튼을 추가해주세요.

요구사항:
- 버튼 텍스트: "초기화" (영어 UI에서는 "Reset")
- 위치: send_btn 오른쪽
- 기능: 채팅 기록 전체 삭제 및 LLM 대화 히스토리 초기화
- 스타일: 기존 버튼과 동일한 스타일 적용
- 확인 다이얼로그 표시 후 초기화 실행
```

### 3.3 역할(Persona) 지정

AI에게 특정 역할을 부여하면 더 전문적인 응답을 받을 수 있습니다.

```
당신은 시니어 PyQt5 개발자입니다.
다음 코드의 메모리 누수 가능성을 검토해주세요.
```

```
당신은 보안 전문가입니다.
이 API 호출 코드에서 보안 취약점을 찾아주세요.
```

### 3.4 단계적 사고 유도

복잡한 문제는 단계적 사고를 요청합니다.

| 키워드 | 사고 수준 | 사용 시기 |
|--------|----------|----------|
| `think` | 기본 분석 | 일반적인 문제 해결 |
| `think hard` | 심층 분석 | 복잡한 로직 설계 |
| `ultrathink` | 최고 수준 분석 | 아키텍처 결정, 성능 최적화 |

```
이 함수의 성능 문제를 ultrathink 수준으로 분석하고
최적화 방안을 3가지 제시해주세요.
```

### 3.5 Few-Shot Learning (예시 제공)

원하는 형식의 예시를 제공합니다.

```
다음 형식으로 함수 문서를 작성해주세요:

예시:
def calculate_area(width: float, height: float) -> float:
    """
    직사각형의 넓이를 계산합니다.

    Args:
        width: 가로 길이 (단위: cm)
        height: 세로 길이 (단위: cm)

    Returns:
        float: 계산된 넓이 (단위: cm²)

    Raises:
        ValueError: 음수 값이 입력된 경우

    Example:
        >>> calculate_area(10, 5)
        50.0
    """
```

### 3.6 제약 조건 명시

명확한 제약 조건을 제시합니다.

```
다음 조건을 반드시 준수해주세요:
- 외부 라이브러리 추가 금지 (기존 import만 사용)
- 기존 함수 시그니처 변경 금지
- 100줄 이내로 구현
- 한국어 주석 필수
- 에러 핸들링 포함
```

---

## 4. 개발 워크플로우

### 4.1 탐색-계획-구현-커밋 (EPIC) 패턴

#### 1단계: 탐색 (Explore)
```
production_app.py에서 TTS 관련 코드를 모두 찾아서 분석해주세요.
코드 작성은 하지 말고 분석만 해주세요.
```

#### 2단계: 계획 (Plan)
```
TTS 기능에 속도 조절 옵션을 추가하려고 합니다.
구현 계획을 단계별로 작성해주세요.
코드는 작성하지 말고 계획만 세워주세요.
```

#### 3단계: 구현 (Implement)
```
위 계획의 1단계를 구현해주세요.
- 파일: production_app.py
- 위치: TTS 설정 부분
```

#### 4단계: 커밋 (Commit)
```
변경사항을 Git에 커밋해주세요.
커밋 메시지는 한글로 작성해주세요.
```

### 4.2 테스트 주도 개발 (TDD)

AI와의 TDD 워크플로우:

```
1. 테스트 먼저 작성
   "다음 입출력을 기반으로 테스트 케이스를 작성해주세요.
    입력: [1, 2, 3, 4, 5]
    출력: 15
    TDD 방식으로 진행합니다. 목(mock) 구현은 하지 마세요."

2. 구현
   "작성된 테스트가 통과하도록 실제 함수를 구현해주세요."

3. 리팩토링
   "테스트가 통과하는 상태에서 코드를 리팩토링해주세요."
```

### 4.3 비주얼 개발 (UI 작업)

```
1. 스크린샷 제공 또는 디자인 설명
2. 구현 요청
3. 결과 스크린샷 확인
4. 2-3회 반복 개선
5. 최종 확인 후 커밋
```

### 4.4 컨텍스트 관리

| 명령 | 용도 |
|------|------|
| `/clear` | 대화 기록 초기화 (새 작업 시작 시) |
| `/compact` | 대화 요약 및 컨텍스트 압축 |

**권장 사항**:
- 새로운 작업 시작 시 `/clear` 실행
- 긴 대화 후 컨텍스트가 부정확해지면 `/clear` 후 재시작
- 관련 없는 이전 대화가 현재 작업에 영향을 주지 않도록 관리

---

## 5. 코드 품질 관리

### 5.1 코드 리뷰 요청

```
다음 코드를 시니어 개발자 관점에서 리뷰해주세요:
- 버그 가능성
- 성능 이슈
- 보안 취약점
- 코드 스타일
- 개선 제안

파일: production_app.py
함수: send_message() (450-520 라인)
```

### 5.2 품질 체크리스트

AI에게 다음 체크리스트 기반 검토를 요청합니다:

```
다음 체크리스트로 코드를 검토해주세요:

□ 에러 핸들링이 적절한가?
□ 엣지 케이스가 처리되었는가?
□ 리소스 정리(cleanup)가 되는가?
□ 하드코딩된 값은 없는가?
□ 로깅이 적절한가?
□ 주석이 충분한가?
□ 네이밍이 명확한가?
□ 중복 코드는 없는가?
```

### 5.3 자동 문서화

```
다음 함수/클래스에 대한 문서를 생성해주세요:
- Docstring (Google 스타일)
- 사용 예시
- 주의사항

단, 기존 코드는 수정하지 마세요.
```

### 5.4 성능 최적화 요청

```
다음 함수의 성능을 분석하고 최적화해주세요:
- 현재 시간 복잡도 분석
- 병목 지점 식별
- 최적화 방안 제시 (최소 2가지)
- 최적화 전후 비교

제약: 기능 동작은 동일하게 유지
```

---

## 6. 버전 관리 및 백업

### 6.1 버전 관리 도구 사용

```bash
# 통합 버전 관리 도구 (권장)
./scripts/version_manager.sh

# 빠른 백업
./scripts/backup.sh

# 롤백
./scripts/rollback.sh
```

### 6.2 Git 커밋 규칙

```
# 커밋 메시지 형식
<타입>: <제목>

<본문>

# 타입 종류
feat:     새로운 기능 추가
fix:      버그 수정
docs:     문서 수정
style:    코드 포맷팅 (기능 변경 없음)
refactor: 코드 리팩토링
test:     테스트 추가/수정
chore:    빌드, 설정 파일 수정
```

### 6.3 백업 전략

| 시점 | 액션 |
|------|------|
| 주요 기능 완료 후 | `./scripts/backup.sh` 실행 |
| 하루 작업 종료 시 | Git 커밋 + 백업 |
| 위험한 변경 전 | 반드시 백업 생성 |
| 대화 중단 가능성 시 | 즉시 백업 |

### 6.4 롤백 시나리오

```bash
# 시나리오 1: 최근 백업으로 복원
./scripts/rollback.sh
# → 목록에서 원하는 버전 선택

# 시나리오 2: Git 특정 커밋으로 복원
git log --oneline -10
git checkout <커밋해시> -- production_app.py

# 시나리오 3: 롤백 취소 (원래대로)
# 롤백 시 자동 생성된 before_rollback 파일 사용
cp backups/production_app_before_rollback_*.py production_app.py
```

---

## 7. 보안 지침

> **중요**: 이 섹션의 모든 지침은 필수 사항입니다. 위반 시 심각한 보안 사고로 이어질 수 있습니다.

### 7.1 민감 정보 절대 금지 사항

```
================================================================================
                    ⚠️  절대 금지 (NEVER DO THIS)  ⚠️
================================================================================

다음 정보는 어떤 상황에서도 코드에 직접 작성하지 마세요:

1. API 키 / 시크릿 키
   - OpenAI API Key (sk-...)
   - Anthropic API Key (sk-ant-...)
   - Google API Key (AIza...)
   - AWS Access Key / Secret Key
   - 기타 모든 서비스 API 키

2. 인증 정보
   - 비밀번호 / 패스워드
   - 토큰 (JWT, OAuth, Bearer 등)
   - SSH 키 / 개인 키
   - 인증서 파일 내용

3. 데이터베이스 정보
   - DB 접속 문자열 (Connection String)
   - DB 사용자명 / 비밀번호
   - 호스트 주소 (프로덕션)

4. 개인 정보
   - 이메일 주소 (테스트용 제외)
   - 전화번호
   - 주소
   - 신용카드 정보

5. 인프라 정보
   - 서버 IP 주소 (프로덕션)
   - 내부 URL / 엔드포인트
   - 방화벽 규칙
================================================================================
```

### 7.2 안전한 민감 정보 관리

#### 7.2.1 환경 변수 사용 (권장)

```python
# ❌ 나쁜 예시 - 절대 금지!
OPENAI_API_KEY = "sk-proj-abc123def456..."
ANTHROPIC_API_KEY = "sk-ant-api03-xyz789..."
DB_PASSWORD = "mySecretPassword123!"

# ✅ 좋은 예시 - 환경 변수 사용
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# 필수 환경 변수 검증
def validate_env_vars():
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {missing}")
```

#### 7.2.2 설정 파일 사용 (.dev.vars)

```python
# ✅ 안전한 설정 파일 로드 함수
def load_env_file(filepath=".dev.vars"):
    """
    .dev.vars 파일에서 환경 변수를 로드합니다.
    이 파일은 절대로 Git에 커밋하면 안 됩니다!
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using system environment variables.")
        return

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
```

#### 7.2.3 .dev.vars 파일 형식

```bash
# .dev.vars 파일 (Git에 절대 커밋 금지!)
# 이 파일은 .gitignore에 반드시 추가되어 있어야 합니다.

# OpenAI API
OPENAI_API_KEY=sk-proj-your-key-here

# Anthropic API
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Google Gemini API
GEMINI_API_KEY=your-gemini-key-here

# Database (프로덕션)
DB_HOST=localhost
DB_USER=app_user
DB_PASSWORD=your-secure-password
```

### 7.3 Git 보안 설정

#### 7.3.1 필수 .gitignore 항목

```gitignore
# ==========================================
# 민감 정보 파일 (필수 제외)
# ==========================================

# 환경 변수 / 설정 파일
.env
.env.*
.dev.vars
*.vars
config.local.*
secrets.*

# API 키 파일
**/api_keys*
**/credentials*
**/secrets*

# 인증서 / 키 파일
*.pem
*.key
*.p12
*.pfx
*.crt
*.cer
id_rsa*
*.ppk

# AWS 자격 증명
.aws/
aws_credentials

# Google Cloud 자격 증명
**/service-account*.json
**/gcloud-*.json

# 데이터베이스
*.sqlite
*.db
*.sql

# 로그 (민감 정보 포함 가능)
*.log
logs/

# IDE 설정 (일부 민감 정보 포함 가능)
.idea/
.vscode/settings.json
*.code-workspace

# 백업 파일
backups/
*.bak
*.backup
```

#### 7.3.2 Git Pre-commit Hook 설정

프로젝트에 자동으로 민감 정보 커밋을 방지하는 hook이 설정되어 있습니다:

```bash
# Hook 위치: .git/hooks/pre-commit
# 실행: 커밋 시 자동으로 민감 정보 검사
```

**검사 대상 패턴:**
- API 키 형식 (sk-, AIza, AKIA 등)
- 비밀번호 변수 (password=, passwd=, pwd= 등)
- 토큰 (token=, bearer, jwt 등)
- 하드코딩된 URL 자격 증명 (://user:pass@)

### 7.4 보안 검사 스크립트 사용

```bash
# 보안 검사 실행
./scripts/security_check.sh

# 검사 항목:
# 1. 하드코딩된 API 키 패턴 검출
# 2. 비밀번호 패턴 검출
# 3. 민감 파일 존재 여부
# 4. .gitignore 설정 확인
```

### 7.5 AI 개발 시 보안 주의사항

#### 7.5.1 Claude Code 사용 시 주의

```
⚠️ Claude Code는 코드 컨텍스트를 Anthropic 서버로 전송합니다.

다음 파일 작업 시 각별히 주의하세요:
- .dev.vars, .env 등 환경 설정 파일
- 프로덕션 API 키가 포함된 설정
- 고객 데이터 또는 개인정보
- 내부 인프라 정보

권장 사항:
1. 민감 파일은 작업 전 임시로 이름 변경
2. 또는 해당 파일 내용 마스킹 후 작업
3. 작업 완료 후 원래 이름으로 복원
```

#### 7.5.2 AI에게 요청 시 주의

```
❌ 나쁜 요청:
"이 API 키 sk-ant-api03-xxx가 작동하는지 확인해줘"
"DB 비밀번호 MySecret123으로 연결하는 코드 작성해줘"

✅ 좋은 요청:
"환경 변수에서 API 키를 읽어오는 코드 작성해줘"
"DB 연결 시 환경 변수를 사용하는 패턴으로 구현해줘"
```

### 7.6 코드 리뷰 보안 체크리스트

```
커밋 전 반드시 확인:

□ 하드코딩된 API 키/비밀번호 없음
□ 주석에 민감 정보 없음
□ 로그에 민감 정보 출력 없음
□ 테스트 코드에 실제 자격 증명 없음
□ 에러 메시지에 민감 정보 노출 없음
□ 설정 파일이 .gitignore에 포함됨
□ URL에 자격 증명 포함 없음
□ 하드코딩된 IP 주소/내부 URL 없음
```

### 7.7 보안 사고 발생 시 대응

```
만약 실수로 민감 정보를 커밋했다면:

1. 즉시 조치 (30분 이내)
   - 해당 API 키/비밀번호 즉시 무효화
   - 새 키 발급

2. Git 히스토리에서 제거
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch 파일명" \
     --prune-empty --tag-name-filter cat -- --all

3. 원격 저장소 강제 푸시
   git push origin --force --all

4. 모든 팀원에게 알림
   - 로컬 저장소 삭제 후 재클론 요청

5. 보안 로그 검토
   - 해당 키로 비정상 접근 있었는지 확인
```

### 7.8 OWASP Top 10 보안 취약점 방지

```
AI에게 코드 보안 검토 요청 시 다음을 포함하세요:

1. Injection (주입 공격)
   - SQL Injection
   - Command Injection
   - LDAP Injection

2. Broken Authentication (인증 결함)
   - 세션 관리
   - 비밀번호 정책

3. Sensitive Data Exposure (민감 데이터 노출)
   - 암호화 여부
   - 전송 보안

4. XML External Entities (XXE)
   - XML 파싱 보안

5. Broken Access Control (접근 제어 실패)
   - 권한 검증

6. Security Misconfiguration (보안 설정 오류)
   - 기본 설정 변경
   - 불필요한 기능 비활성화

7. Cross-Site Scripting (XSS)
   - 입력값 이스케이프
   - 출력 인코딩

8. Insecure Deserialization (안전하지 않은 역직렬화)
   - 신뢰할 수 없는 데이터 역직렬화 금지

9. Using Components with Known Vulnerabilities
   - 라이브러리 버전 관리
   - 보안 패치 적용

10. Insufficient Logging & Monitoring
    - 보안 이벤트 로깅
    - 이상 탐지
```

---

## 8. 문제 해결 가이드

### 8.1 AI 응답이 부정확할 때

```
1. 컨텍스트 초기화
   /clear 실행 후 필요한 정보만 다시 제공

2. 더 구체적인 프롬프트 작성
   - 파일 경로 명시
   - 라인 번호 지정
   - 예상 결과 설명

3. 단계 분할
   복잡한 요청을 여러 단계로 나눔

4. 다른 접근 방식 시도
   "다른 방법으로 구현해주세요" 요청
```

### 8.2 대화 중단 복구

```bash
# 1. 프로젝트 상태 확인
ls -la /home/orangepi/dual_npu_demo/

# 2. 개발 로그 확인
cat DEVLOG.md

# 3. Git 상태 확인
git status
git log --oneline -5

# 4. 새 대화에서 컨텍스트 복원
# (DEVLOG.md 하단의 프롬프트 템플릿 사용)
```

### 8.3 코드 충돌 해결

```
Git 충돌 발생 시:
1. 충돌 파일 확인: git status
2. 충돌 내용 분석 요청
3. 해결 방안 제시 요청
4. 수동 확인 후 적용
5. git add 및 커밋
```

### 8.4 일반적인 오류 대응

| 오류 유형 | 대응 방법 |
|----------|----------|
| Import 오류 | 필요한 패키지 설치 확인 요청 |
| 문법 오류 | 해당 라인 컨텍스트와 함께 수정 요청 |
| 런타임 오류 | 전체 traceback 제공 후 분석 요청 |
| 로직 오류 | 예상 vs 실제 결과 비교 설명 |

---

## 9. 대화 연속성 유지

### 9.1 대화 종료 전 체크리스트

```
□ 현재 작업 상태 DEVLOG.md에 기록
□ 미완료 작업 TODO로 명시
□ 백업 생성 (./scripts/backup.sh)
□ Git 커밋 (중요 변경사항)
□ 다음 작업 계획 기록
```

### 9.2 새 대화 시작 프롬프트 템플릿

```
이전 개발 대화가 중단되어 새로 시작합니다.

프로젝트: Dual NPU Demo Application
위치: /home/orangepi/dual_npu_demo/

다음 파일들을 읽어서 이전 개발 내용을 파악해주세요:
1. DEVLOG.md - 개발 로그 (필수)
2. README.md - 프로젝트 개요
3. DEV_GUIDELINES.md - 개발 지침

그 후 다음 작업을 진행해주세요:
[여기에 요청사항 작성]
```

### 9.3 작업 인수인계 문서화

주요 작업 완료 시 다음 형식으로 DEVLOG.md 업데이트:

```markdown
### [YYYY-MM-DD] 작업 내용

#### 완료 사항
- 기능 A 구현 완료
- 버그 B 수정

#### 변경 파일
- production_app.py: 450-520 라인 수정
- utils.py: 새 함수 추가

#### 미완료/다음 작업
- [ ] 기능 C 구현 필요
- [ ] 테스트 코드 작성

#### 주의사항
- 설정 X 변경 시 재시작 필요
- 함수 Y는 deprecated 예정
```

---

## 10. 참고 자료

### 10.1 공식 문서

- [Claude Code Best Practices (Anthropic)](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Prompt Engineering Guide (OpenAI)](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### 10.2 추천 학습 자료

- [Google Cloud: AI Coding Assistants Best Practices](https://cloud.google.com/blog/topics/developers-practitioners/five-best-practices-for-using-ai-coding-assistants)
- [JetBrains: Coding Guidelines for AI Agents](https://blog.jetbrains.com/idea/2025/05/coding-guidelines-for-your-ai-agents/)
- [The Ultimate AI Coding Guide](https://www.sabrina.dev/p/ultimate-ai-coding-guide-claude-code)

### 10.3 프로젝트 관련 문서

| 문서 | 위치 | 설명 |
|------|------|------|
| README.md | 프로젝트 루트 | 프로젝트 개요 |
| DEVLOG.md | 프로젝트 루트 | 개발 로그 |
| VISION_LLM_OPTIONS.md | 프로젝트 루트 | LLM 옵션 설명 |

---

## 부록: 빠른 참조 카드

### 프롬프트 치트시트

```
# 탐색
"~에 대해 분석해줘. 코드는 작성하지 마."

# 계획
"~를 구현하려면 어떤 단계가 필요해? 계획만 세워줘."

# 구현
"위 계획의 N단계를 구현해줘."

# 리뷰
"이 코드를 시니어 개발자 관점에서 리뷰해줘."

# 디버깅
"이 에러의 원인을 분석하고 해결책을 제시해줘: [에러 메시지]"

# 최적화
"이 함수의 성능을 분석하고 개선해줘."

# 문서화
"이 코드에 상세한 주석을 추가해줘."
```

### 필수 명령어

```bash
# 백업
./scripts/backup.sh

# 롤백
./scripts/rollback.sh

# 버전 관리 (메뉴)
./scripts/version_manager.sh

# 앱 실행
./run_production.sh

# Git 상태
git status && git log --oneline -5
```

---

> **문서 관리**: 이 문서는 개발 경험에 따라 지속적으로 업데이트됩니다.
>
> **마지막 업데이트**: 2025년 12월 28일
