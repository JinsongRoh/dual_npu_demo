# Vision LLM 옵션 조사 결과 (2025년 12월)

## 개요

YOLOX-S로 감지한 화면을 분석할 수 있는 Vision LLM 옵션들을 조사했습니다.

---

## 1. 무료/저렴한 클라우드 API 옵션

### 1.1 Google Gemini API (추천 - 무료)

| 항목 | 내용 |
|------|------|
| **모델** | Gemini 2.5 Flash, Gemini 2.0 Flash |
| **무료 한도** | 15 RPM, 1,500 요청/일, 신용카드 불필요 |
| **이미지 비용** | $0.0011/이미지 (560 토큰) |
| **장점** | 가장 관대한 무료 티어, 멀티모달 네이티브 |
| **Vision 성능** | 텍스트+이미지+비디오 동시 처리 |

```python
# Gemini API 사용 예시
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content([image, "이 이미지에서 무엇이 보이나요?"])
```

**참고 링크:**
- [Gemini API 가격](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini 무료 티어 가이드](https://dev.to/claudeprime/gemini-20-flash-api-free-tier-guide-for-developers-4bh2)

---

### 1.2 Groq API (무료 - 빠름)

| 항목 | 내용 |
|------|------|
| **모델** | Llama 4 Scout/Maverick, LLaVA v1.5 7B |
| **무료 한도** | 개발자 무료 티어 제공 |
| **속도** | GPT-4o 대비 4배 이상 빠름 |
| **장점** | 초저지연, 무료 티어 |
| **Vision 기능** | VQA, 캡션 생성, OCR |

```python
# Groq Vision API 사용 예시
from groq import Groq
client = Groq(api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="llava-v1.5-7b-4096-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": "이 이미지를 분석해주세요"}
        ]
    }]
)
```

**참고 링크:**
- [Groq Vision Docs](https://console.groq.com/docs/vision)
- [LLaVA on Groq](https://groq.com/blog/introducing-llava-v1-5-7b-on-groqcloud-unlocking-the-power-of-multimodal-ai)

---

### 1.3 MiniMax API (저렴)

| 항목 | 내용 |
|------|------|
| **모델** | MiniMax M2, M1 |
| **가격** | $0.15/1M 입력, $0.60/1M 출력 (GPT-4 대비 1/60) |
| **무료 티어** | 신규 가입시 토큰 제공 |
| **장점** | 매우 저렴, 멀티모달 지원 |
| **Vision 기능** | 스크린샷 분석, OCR, UI 분석 |

**참고 링크:**
- [MiniMax 공식](https://www.minimax.io/)
- [MiniMax 무료 사용법](https://apidog.com/blog/how-to-use-minimax-m2-for-free/)

---

### 1.4 Claude API (유료 - 고성능)

| 항목 | 내용 |
|------|------|
| **모델** | Claude 4 Sonnet, Claude 4 Opus |
| **가격** | Sonnet: $3/1M 입력, $15/1M 출력 |
| **장점** | 뛰어난 추론 능력, 한국어 지원 |
| **Vision 기능** | 이미지 이해 및 코드 생성 |

```python
# Claude Vision API 사용 예시
import anthropic
client = anthropic.Anthropic(api_key="YOUR_API_KEY")
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
            {"type": "text", "text": "이 이미지에서 무엇이 보이나요?"}
        ]
    }]
)
```

**참고 링크:**
- [Claude API Docs](https://docs.anthropic.com/en/docs/vision)

---

## 2. 로컬 실행 옵션 (RK3588 NPU)

### 2.1 Qwen2-VL / Qwen3-VL (추천 - 완전 무료)

| 항목 | 내용 |
|------|------|
| **모델** | Qwen2-VL-2B-Instruct, Qwen3-VL-2B |
| **성능** | 15.39 토큰/초 (RK3588 NPU) |
| **비용** | 완전 무료 (로컬 실행) |
| **장점** | 인터넷 불필요, 프라이버시 보장 |
| **지원** | RKLLM 공식 지원 |

**설치 방법:**
```bash
# 모델 다운로드
git lfs install
git clone https://www.modelscope.cn/radxa/Qwen2-VL-2B-RKLLM.git

# 실행
./demo image.jpg ./qwen2_vl_2b_vision_rk3588.rknn ./qwen2-vl-llm_rk3588.rkllm 128 512 3
```

**참고 링크:**
- [Radxa RKLLM Qwen2-VL](https://docs.radxa.com/en/rock5/rock5b/app-development/rkllm_qwen2_vl)
- [GitHub rknn-llm](https://github.com/airockchip/rknn-llm)
- [Qwen3-VL-2B-NPU](https://github.com/Qengineering/Qwen3-VL-2B-NPU)

---

## 3. 비용 비교표

| 서비스 | 입력 (1M 토큰) | 출력 (1M 토큰) | 무료 티어 | Vision |
|--------|---------------|---------------|-----------|--------|
| **Gemini 2.5 Flash** | $0.15 | $0.60 | 1,500요청/일 | O |
| **Groq LLaVA** | 무료 | 무료 | O | O |
| **MiniMax M2** | $0.15 | $0.60 | 토큰 지급 | O |
| **Claude Sonnet** | $3.00 | $15.00 | X | O |
| **Qwen2-VL (로컬)** | 무료 | 무료 | 완전무료 | O |

---

## 4. 권장 구현 방안

### 4.1 1단계: 무료 옵션 먼저 구현

1. **Gemini API** - 무료 티어로 시작 (추천)
2. **Groq API** - 빠른 응답 필요시
3. **RK3588 Qwen2-VL** - 오프라인 필요시

### 4.2 2단계: 유료 옵션 추가

1. **Claude API** - 고품질 분석 필요시
2. **MiniMax** - 비용 효율적 대안

### 4.3 UI 구현

```
[ 모델 선택 ▼ ]
├─ Gemini 2.5 Flash (무료)
├─ Groq LLaVA (무료/빠름)
├─ Claude Sonnet (유료/고품질)
├─ MiniMax M2 (저렴)
└─ Qwen2-VL 로컬 (무료/오프라인)
```

### 4.4 YOLO 화면 전송 방식

```python
# 1. YOLO 감지된 프레임 캡처
frame_with_boxes = detector.detect(frame)[1]

# 2. Base64 인코딩
_, buffer = cv2.imencode('.jpg', frame_with_boxes)
base64_image = base64.b64encode(buffer).decode('utf-8')

# 3. Vision LLM에 전송
response = vision_llm.analyze(base64_image, user_question)
```

---

## 5. API 키 설정

환경 변수 또는 설정 파일로 관리:

```bash
# ~/.bashrc 또는 .env 파일
export GEMINI_API_KEY="your_gemini_key"
export GROQ_API_KEY="your_groq_key"
export ANTHROPIC_API_KEY="your_claude_key"
export MINIMAX_API_KEY="your_minimax_key"
```

---

## 6. 참고 자료

### 공식 문서
- [Gemini API](https://ai.google.dev/gemini-api/docs)
- [Groq Vision](https://console.groq.com/docs/vision)
- [Claude Vision](https://docs.anthropic.com/en/docs/vision)
- [RKLLM Docs](https://docs.radxa.com/en/rock5/rock5b/app-development/rkllm_usage)

### 가격 비교
- [LLM API Pricing 2025](https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025)
- [Vision LLM Top 10](https://www.datacamp.com/blog/top-vision-language-models)

### GitHub 저장소
- [rknn-llm](https://github.com/airockchip/rknn-llm)
- [Qwen3-VL-2B-NPU](https://github.com/Qengineering/Qwen3-VL-2B-NPU)

---

*작성일: 2025년 12월 28일*
