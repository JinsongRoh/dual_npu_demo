#!/usr/bin/env python3
"""
================================================================================
RK3588 Local VLM API Server (RK3588 로컬 VLM API 서버)
================================================================================
회사명: MetaVu Co., Ltd.
개발자: JINSONG ROH
이메일: enjoydays@metavu.io
홈페이지: www.metavu.io
저작권: © 2025 MetaVu Co., Ltd. All rights reserved.

설명:
    RK3588 NPU를 활용한 로컬 Vision-Language Model (VLM) API 서버
    OpenAI 호환 API 엔드포인트 제공

지원 모델:
    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL
    - MiniCPM-V
    - InternVL2 / InternVL3
    - Janus-Pro
    - SmolVLM
    - DeepSeekOCR

요구사항:
    - RK3588/RK3588S/RK3576 SoC
    - RKNPU Driver >= 0.9.7
    - RKLLM Runtime >= 1.2.0
================================================================================
"""

import os
import sys
import json
import base64
import time
import threading
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# FastAPI 서버
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[Warning] FastAPI not installed. Run: pip3 install fastapi uvicorn")

# =============================================================================
# 모델 정의 (Model Definitions)
# =============================================================================

class VLMModelType(Enum):
    """지원되는 VLM 모델 타입"""
    QWEN2_VL = "qwen2-vl"
    QWEN25_VL = "qwen2.5-vl"
    QWEN3_VL = "qwen3-vl"
    MINICPM_V = "minicpm-v"
    INTERNVL2 = "internvl2"
    INTERNVL3 = "internvl3"
    JANUS_PRO = "janus-pro"
    SMOLVLM = "smolvlm"
    DEEPSEEK_OCR = "deepseek-ocr"


# 모델 설정 정보
VLM_MODEL_CONFIGS = {
    "qwen2-vl-2b": {
        "type": VLMModelType.QWEN2_VL,
        "name": "Qwen2-VL-2B",
        "description": "Alibaba Qwen2-VL 2B - 빠른 추론",
        "rkllm_file": "qwen2-vl-2b-w8a8_rk3588.rkllm",
        "vision_file": "qwen2_vl_2b_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 512,
    },
    "qwen2.5-vl-3b": {
        "type": VLMModelType.QWEN25_VL,
        "name": "Qwen2.5-VL-3B",
        "description": "Alibaba Qwen2.5-VL 3B - 균형잡힌 성능",
        "rkllm_file": "qwen2.5-vl-3b-w8a8_rk3588.rkllm",
        "vision_file": "qwen2_5_vl_3b_vision_rk3588.rknn",
        "context_length": 4096,
        "max_tokens": 1024,
    },
    "qwen3-vl-2b": {
        "type": VLMModelType.QWEN3_VL,
        "name": "Qwen3-VL-2B",
        "description": "Alibaba Qwen3-VL 2B - 최신 모델",
        "rkllm_file": "qwen3-vl-2b-w8a8_rk3588.rkllm",
        "vision_file": "qwen3_vl_2b_vision_rk3588.rknn",
        "context_length": 4096,
        "max_tokens": 1024,
    },
    "minicpm-v-2.6": {
        "type": VLMModelType.MINICPM_V,
        "name": "MiniCPM-V-2.6",
        "description": "OpenBMB MiniCPM-V 2.6 - 경량 모델",
        "rkllm_file": "minicpm-v-2.6-w8a8_rk3588.rkllm",
        "vision_file": "minicpm_v_2.6_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 512,
    },
    "internvl2-1b": {
        "type": VLMModelType.INTERNVL2,
        "name": "InternVL2-1B",
        "description": "Shanghai AI Lab InternVL2 1B",
        "rkllm_file": "internvl2-1b-w8a8_rk3588.rkllm",
        "vision_file": "internvl2_1b_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 512,
    },
    "internvl3-1b": {
        "type": VLMModelType.INTERNVL3,
        "name": "InternVL3-1B",
        "description": "Shanghai AI Lab InternVL3 1B - 최신",
        "rkllm_file": "internvl3-1b-w8a8_rk3588.rkllm",
        "vision_file": "internvl3_1b_vision_rk3588.rknn",
        "context_length": 4096,
        "max_tokens": 1024,
    },
    "janus-pro-1b": {
        "type": VLMModelType.JANUS_PRO,
        "name": "Janus-Pro-1B",
        "description": "DeepSeek Janus-Pro 1B - 다목적",
        "rkllm_file": "janus-pro-1b-w8a8_rk3588.rkllm",
        "vision_file": "janus_pro_1b_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 512,
    },
    "smolvlm-instruct": {
        "type": VLMModelType.SMOLVLM,
        "name": "SmolVLM-Instruct",
        "description": "HuggingFace SmolVLM - 초경량",
        "rkllm_file": "smolvlm-instruct-w8a8_rk3588.rkllm",
        "vision_file": "smolvlm_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 256,
    },
    "deepseek-ocr": {
        "type": VLMModelType.DEEPSEEK_OCR,
        "name": "DeepSeek-OCR",
        "description": "DeepSeek OCR 특화 모델",
        "rkllm_file": "deepseek-ocr-w8a8_rk3588.rkllm",
        "vision_file": "deepseek_ocr_vision_rk3588.rknn",
        "context_length": 2048,
        "max_tokens": 1024,
    },
}

# =============================================================================
# RKLLM 런타임 래퍼 (RKLLM Runtime Wrapper)
# =============================================================================

class RKLLMRuntime:
    """
    RKLLM 런타임 래퍼 클래스

    RKLLM C/C++ 라이브러리를 Python에서 사용하기 위한 래퍼
    실제 RKLLM이 설치되지 않은 경우 시뮬레이션 모드로 동작
    """

    def __init__(self, model_dir: str = "/home/orangepi/rkllm_models"):
        """
        Args:
            model_dir: RKLLM 모델 파일이 저장된 디렉토리
        """
        self.model_dir = model_dir
        self.loaded_model = None
        self.rkllm_available = self._check_rkllm_available()

        # RKLLAMA 서버 URL (설치된 경우)
        self.rkllama_url = os.environ.get("RKLLAMA_URL", "http://localhost:8080")

    def _check_rkllm_available(self) -> bool:
        """RKLLM 런타임 사용 가능 여부 확인"""
        try:
            # RKLLM 라이브러리 확인
            result = subprocess.run(
                ["which", "rkllama_server"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[RKLLM] RKLLAMA server found")
                return True

            # librkllm.so 확인
            lib_paths = [
                "/usr/lib/librkllmrt.so",
                "/usr/local/lib/librkllmrt.so",
                "/opt/rkllm/lib/librkllmrt.so"
            ]
            for path in lib_paths:
                if os.path.exists(path):
                    print(f"[RKLLM] Runtime library found: {path}")
                    return True

            print("[RKLLM] Runtime not found - running in simulation mode")
            return False
        except Exception as e:
            print(f"[RKLLM] Check failed: {e}")
            return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        available = []

        for model_id, config in VLM_MODEL_CONFIGS.items():
            # 모델 파일 존재 여부 확인
            rkllm_path = os.path.join(self.model_dir, config["rkllm_file"])
            vision_path = os.path.join(self.model_dir, config["vision_file"])

            model_info = {
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "type": config["type"].value,
                "available": os.path.exists(rkllm_path) and os.path.exists(vision_path),
                "rkllm_path": rkllm_path,
                "vision_path": vision_path,
            }
            available.append(model_info)

        return available

    def load_model(self, model_id: str) -> bool:
        """모델 로드"""
        if model_id not in VLM_MODEL_CONFIGS:
            print(f"[RKLLM] Unknown model: {model_id}")
            return False

        config = VLM_MODEL_CONFIGS[model_id]
        self.loaded_model = model_id
        print(f"[RKLLM] Model loaded: {config['name']}")
        return True

    def generate(self, image_base64: str, prompt: str,
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        이미지와 프롬프트로 응답 생성

        Args:
            image_base64: Base64 인코딩된 이미지
            prompt: 사용자 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도

        Returns:
            생성된 텍스트 응답
        """
        if not self.loaded_model:
            return "Error: No model loaded"

        config = VLM_MODEL_CONFIGS.get(self.loaded_model, {})

        # RKLLAMA 서버가 실행 중인 경우 API 호출
        if self.rkllm_available:
            return self._call_rkllama(image_base64, prompt, max_tokens, temperature)

        # 시뮬레이션 모드 (RKLLM 미설치 시)
        return self._simulate_response(image_base64, prompt, config)

    def _call_rkllama(self, image_base64: str, prompt: str,
                      max_tokens: int, temperature: float) -> str:
        """RKLLAMA API 호출"""
        try:
            import requests

            url = f"{self.rkllama_url}/v1/chat/completions"

            payload = {
                "model": self.loaded_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                return f"RKLLAMA Error: {response.status_code}"

        except requests.exceptions.ConnectionError:
            return "Error: RKLLAMA server not running. Start with: rkllama_server --models /path/to/models"
        except Exception as e:
            return f"RKLLAMA Error: {str(e)}"

    def _simulate_response(self, image_base64: str, prompt: str,
                          config: Dict[str, Any]) -> str:
        """RKLLM 미설치 시 시뮬레이션 응답"""
        model_name = config.get("name", "Unknown")
        return (
            f"[시뮬레이션 모드 - {model_name}]\n\n"
            f"RKLLM 런타임이 설치되지 않았습니다.\n\n"
            f"설치 방법:\n"
            f"1. RKLLAMA 설치: pip install rkllama\n"
            f"2. 모델 다운로드: rkllama_client pull <model>\n"
            f"3. 서버 시작: rkllama_server --models ~/rkllm_models\n\n"
            f"요청된 프롬프트: {prompt[:100]}..."
        )


# =============================================================================
# FastAPI 서버 (API Server)
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="RK3588 Local VLM API",
        description="RK3588 NPU 기반 로컬 Vision-Language Model API 서버",
        version="1.0.0"
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 전역 런타임 인스턴스
    rkllm_runtime = RKLLMRuntime()

    # Pydantic 모델 정의
    class ImageURL(BaseModel):
        url: str

    class ContentItem(BaseModel):
        type: str
        text: Optional[str] = None
        image_url: Optional[ImageURL] = None

    class Message(BaseModel):
        role: str
        content: Any  # str or List[ContentItem]

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[Message]
        max_tokens: Optional[int] = 512
        temperature: Optional[float] = 0.7
        stream: Optional[bool] = False

    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[Dict[str, Any]]
        usage: Dict[str, int]

    # API 엔드포인트
    @app.get("/")
    async def root():
        """서버 상태 확인"""
        return {
            "status": "running",
            "server": "RK3588 Local VLM API",
            "version": "1.0.0",
            "rkllm_available": rkllm_runtime.rkllm_available
        }

    @app.get("/v1/models")
    async def list_models():
        """사용 가능한 모델 목록"""
        models = rkllm_runtime.get_available_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "name": m["name"],
                    "description": m["description"],
                    "available": m["available"]
                }
                for m in models
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI 호환 Chat Completions API"""

        # 모델 로드
        if not rkllm_runtime.load_model(request.model):
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}"
            )

        # 메시지에서 이미지와 텍스트 추출
        image_base64 = None
        prompt_text = ""

        for msg in request.messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    prompt_text = msg.content
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prompt_text = item.get("text", "")
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                                if url.startswith("data:image"):
                                    # Base64 이미지
                                    image_base64 = url.split(",")[1] if "," in url else url

        if not image_base64:
            raise HTTPException(
                status_code=400,
                detail="No image provided in request"
            )

        # 응답 생성
        response_text = rkllm_runtime.generate(
            image_base64=image_base64,
            prompt=prompt_text,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt_text.split()) + len(response_text.split())
            }
        )

    @app.get("/health")
    async def health_check():
        """헬스 체크"""
        return {
            "status": "healthy",
            "rkllm_available": rkllm_runtime.rkllm_available,
            "loaded_model": rkllm_runtime.loaded_model
        }


# =============================================================================
# 메인 실행 (Main)
# =============================================================================

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="RK3588 Local VLM API Server")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8088, help="서버 포트 (기본: 8088)")
    parser.add_argument("--model-dir", default="/home/orangepi/rkllm_models",
                       help="모델 디렉토리 경로")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed")
        print("Install with: pip3 install fastapi uvicorn")
        sys.exit(1)

    # 모델 디렉토리 설정
    global rkllm_runtime
    rkllm_runtime = RKLLMRuntime(model_dir=args.model_dir)

    print(f"""
================================================================================
         RK3588 Local VLM API Server
================================================================================
  Host: {args.host}
  Port: {args.port}
  Model Dir: {args.model_dir}
  RKLLM Available: {rkllm_runtime.rkllm_available}

  API Endpoints:
    - GET  /           : 서버 상태
    - GET  /v1/models  : 모델 목록
    - POST /v1/chat/completions : Chat API (OpenAI 호환)
    - GET  /health     : 헬스 체크
================================================================================
""")

    # 서버 시작
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info"
    )


if __name__ == "__main__":
    main()
