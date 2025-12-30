#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Production Dual NPU Application (프로덕션 듀얼 NPU 애플리케이션)
================================================================================

이 프로그램은 두 개의 NPU를 동시에 활용하는 데모 애플리케이션입니다:

1. DX-M1 NPU (DeepX):
   - YOLOX-S 모델을 사용한 실시간 객체 감지
   - 최대 10명까지 다중 인식 지원
   - 각 객체별 고유 색상 바운딩 박스

2. RK3588 NPU (Rockchip):
   - Vision LLM 채팅 기능
   - 다양한 LLM 프로바이더 지원 (Gemini, GPT-4o, Claude 등)
   - STT/TTS 음성 입출력 지원

주요 기능:
- 실시간 카메라 영상 처리 및 객체 감지
- AI 비전 어시스턴트 채팅
- 실시간 통역 모드 (다국어 번역)
- 자동 장면 설명 기능
- UI 다국어 지원 (10개 언어)
- 시스템 리소스 모니터링

개발 환경:
- Orange Pi 5B (RK3588)
- DeepX DX-M1 NPU 가속기
- Python 3.x + PyQt5

================================================================================
개발 정보 (Development Info)
================================================================================
회사명: MetaVu Co., Ltd.
개발자: JINSONG ROH
이메일: enjoydays@metavu.io
홈페이지: www.metavu.io
저작권: © 2025 MetaVu Co., Ltd. All rights reserved.
최종수정: 2025년 12월 28일 오후 7시 42분
================================================================================
"""

# =============================================================================
# 라이브러리 임포트 (Library Imports)
# =============================================================================

import sys
import os

# -----------------------------------------------------------------------------
# Qt 플러그인 환경 설정
# PyQt5와 OpenCV의 Qt 라이브러리 충돌을 방지하기 위한 설정
# 반드시 다른 Qt 관련 임포트 전에 설정해야 함
# -----------------------------------------------------------------------------
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms'
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # X11 윈도우 시스템 사용

# -----------------------------------------------------------------------------
# PyQt5 GUI 컴포넌트
# -----------------------------------------------------------------------------
from PyQt5.QtWidgets import (
    QApplication,      # 애플리케이션 인스턴스
    QMainWindow,       # 메인 윈도우
    QWidget,           # 기본 위젯
    QVBoxLayout,       # 수직 레이아웃
    QHBoxLayout,       # 수평 레이아웃
    QLabel,            # 텍스트/이미지 라벨
    QTextEdit,         # 멀티라인 텍스트 편집기
    QPushButton,       # 푸시 버튼
    QFrame,            # 프레임 컨테이너
    QGroupBox,         # 그룹 박스
    QLineEdit,         # 단일라인 텍스트 입력
    QProgressBar,      # 진행 바
    QComboBox,         # 드롭다운 선택 박스
    QSpinBox,          # 숫자 입력 스핀박스
    QCheckBox,         # 체크박스
    QDialog,           # 다이얼로그 창
    QDialogButtonBox,  # 다이얼로그 버튼 박스
    QSlider            # 슬라이더
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# -----------------------------------------------------------------------------
# 표준 라이브러리 및 서드파티 라이브러리
# -----------------------------------------------------------------------------
import base64                                    # Base64 인코딩/디코딩
import cv2                                       # OpenCV - 영상 처리
import numpy as np                               # NumPy - 수치 연산
import requests                                  # HTTP 요청
import threading                                 # 스레딩
import time                                      # 시간 관련 함수
import struct                                    # 바이너리 데이터 처리
import tempfile                                  # 임시 파일 처리
import io                                        # I/O 스트림
from collections import Counter, OrderedDict, deque  # 컬렉션 유틸리티
from datetime import datetime                    # 날짜/시간 처리

# -----------------------------------------------------------------------------
# 오디오 라이브러리 (STT/TTS 기능용)
# 설치되지 않은 경우 음성 기능 비활성화
# -----------------------------------------------------------------------------
try:
    import sounddevice as sd    # 오디오 녹음/재생
    import soundfile as sf      # 오디오 파일 읽기/쓰기
    from openai import OpenAI   # OpenAI API (Whisper STT, TTS)
    # Set default audio devices by NAME (indices change after reboot!)
    # Input: USB LifeCam microphone, Output: HDMI1 via rockchip
    sd.default.device = ('LifeCam', 'rockchip-hdmi1')  # Full name to avoid dmix conflict
    sd.default.samplerate = 48000
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[Audio] sounddevice/openai not available - 음성 기능 비활성화")

# -----------------------------------------------------------------------------
# DeepX SDK 경로 추가
# DX-M1 NPU를 사용하기 위한 Python 패키지 경로
# -----------------------------------------------------------------------------
sys.path.insert(0, '/home/orangepi/deepx_sdk/dx_rt/python_package/src')

# =============================================================================
# 전역 설정 (Global Configuration)
# =============================================================================

# RK3588 로컬 LLM API 엔드포인트 (현재 미사용)
RKLLAMA_API = "http://localhost:8080"

# -----------------------------------------------------------------------------
# Vision LLM 모델 설정
# 지원하는 클라우드 비전 LLM 모델 목록
# provider: API 제공자, model: 모델 ID, cost: 비용 등급
# -----------------------------------------------------------------------------
VISION_LLM_OPTIONS = {
    # =============================================
    # 클라우드 LLM 모델 (Cloud LLM Models)
    # =============================================
    "Gemini Flash (Free)": {"provider": "gemini", "model": "gemini-2.0-flash", "cost": "Free"},
    "Gemini Pro": {"provider": "gemini", "model": "gemini-1.5-pro", "cost": "Paid"},
    "GPT-4o Vision": {"provider": "openai", "model": "gpt-4o", "cost": "Paid"},
    "GPT-4o Mini": {"provider": "openai", "model": "gpt-4o-mini", "cost": "Cheap"},
    "Claude Sonnet": {"provider": "claude", "model": "claude-sonnet-4-20250514", "cost": "Paid"},
    "Claude Haiku": {"provider": "claude", "model": "claude-3-haiku-20240307", "cost": "Cheap"},
    # =============================================
    # RK3588 로컬 VLM 모델 (Local VLM Models)
    # =============================================
    "🖥️ Qwen2.5-VL-3B": {"provider": "local_vlm", "model": "qwen2.5-vl-3b", "cost": "Local"},
}

# RK3588 로컬 VLM API 서버 설정
LOCAL_VLM_API_URL = os.environ.get("LOCAL_VLM_API_URL", "http://localhost:8088")

# =============================================================================
# RK3588 로컬 VLM 클래스 (Direct NPU Inference)
# =============================================================================

# =============================================================================
# RK3588 VLM 모델 설정 (다중 모델 지원)
# =============================================================================
RK3588_VLM_MODELS = {
    # RKLLM 1.2.1 호환 모델만 지원
    "qwen2.5-vl-3b": {
        "path": "/mnt/external/rkllm_models/Qwen2.5-VL-3B",
        "vision": "vision_encoder.rknn",
        "llm": "language_model_w8a8.rkllm",
        "image_size": 476,
        "img_start": "<|vision_start|>",
        "img_end": "<|vision_end|>",
        "img_content": "<|image_pad|>",
    },
    # qwen2-vl-2b: RKLLM 1.1.4용으로 1.2.1과 호환 불가
    # minicpm-v-2.6: 8GB LLM으로 메모리 부족
}

# 기본 모델 설정
RK3588_DEFAULT_MODEL = "qwen2.5-vl-3b"

class RK3588LocalVLM:
    """
    RK3588 NPU에서 직접 VLM 추론을 수행하는 클래스 (싱글톤)

    지원 모델:
    - qwen2.5-vl-3b: Vision 4.5초, LLM ~8 tokens/s
    - qwen2-vl-2b: Vision 3초, LLM ~12 tokens/s
    """
    _instance = None
    _initialized = False
    _current_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if RK3588LocalVLM._initialized:
            return

        self.vision_session = None
        self.rk_llm = None
        self.is_ready = False
        self.response_text = ""
        self.IMAGE_HEIGHT = 476
        self.IMAGE_WIDTH = 476
        self.current_model = None
        self.model_config = None

    def initialize(self, model_name=None):
        """
        모델 초기화 (처음 호출 시에만 실행)

        Args:
            model_name: 모델 이름 (qwen2.5-vl-3b, qwen2-vl-2b 등)
        """
        if model_name is None:
            model_name = RK3588_DEFAULT_MODEL

        # 이미 같은 모델이 로드되어 있으면 스킵
        if self.is_ready and self.current_model == model_name:
            return True

        # 다른 모델 요청 시 경고 (모델 교체 미지원)
        if self.is_ready and self.current_model != model_name:
            print(f"[RK3588-VLM] 경고: 이미 {self.current_model} 로드됨. 재시작 필요.")
            return True  # 기존 모델로 계속 진행

        # 모델 설정 확인
        if model_name not in RK3588_VLM_MODELS:
            print(f"[RK3588-VLM] 오류: {model_name} 모델 설정 없음")
            return False

        self.model_config = RK3588_VLM_MODELS[model_name]
        model_path = self.model_config["path"]
        vision_file = f"{model_path}/{self.model_config['vision']}"
        llm_file = f"{model_path}/{self.model_config['llm']}"

        # 이미지 크기 설정
        self.IMAGE_HEIGHT = self.model_config.get("image_size", 476)
        self.IMAGE_WIDTH = self.model_config.get("image_size", 476)

        try:
            print(f"[RK3588-VLM] {model_name} 모델 초기화 시작...")

            # Vision Encoder 로드 (RKNN)
            import ztu_somemodelruntime_rknnlite2 as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 3
            self.vision_session = ort.InferenceSession(vision_file, sess_options)
            self.vision_input_name = self.vision_session.get_inputs()[0].name
            self.vision_output_name = self.vision_session.get_outputs()[0].name
            print(f"[RK3588-VLM] Vision Encoder 로드 완료")

            # LLM 로드 (RKLLM)
            from rkllm_binding import (
                RKLLMRuntime, RKLLMParam, RKLLMInput, RKLLMInferParam,
                LLMCallState, RKLLMInputType, RKLLMInferMode
            )
            self.RKLLMInput = RKLLMInput
            self.RKLLMInferParam = RKLLMInferParam
            self.RKLLMInputType = RKLLMInputType
            self.RKLLMInferMode = RKLLMInferMode
            self.LLMCallState = LLMCallState

            self.rk_llm = RKLLMRuntime()
            param = self.rk_llm.create_default_param()
            param.model_path = llm_file.encode('utf-8')
            param.top_k = 1
            param.max_new_tokens = 256
            param.max_context_len = 512
            param.skip_special_token = True
            param.img_start = self.model_config.get("img_start", "<|vision_start|>").encode('utf-8')
            param.img_end = self.model_config.get("img_end", "<|vision_end|>").encode('utf-8')
            param.img_content = self.model_config.get("img_content", "<|image_pad|>").encode('utf-8')
            param.extend_param.base_domain_id = 1

            self.rk_llm.init(param, self._llm_callback)
            self.rk_llm.set_chat_template(
                system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                prompt_prefix="<|im_start|>user\n",
                prompt_postfix="<|im_end|>\n<|im_start|>assistant\n"
            )
            print(f"[RK3588-VLM] LLM 로드 완료")

            self.is_ready = True
            self.current_model = model_name
            RK3588LocalVLM._initialized = True
            RK3588LocalVLM._current_model = model_name
            print(f"[RK3588-VLM] {model_name} 모델 초기화 완료!")
            return True

        except Exception as e:
            print(f"[RK3588-VLM] 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _llm_callback(self, result_ptr, userdata_ptr, state_enum):
        """LLM 응답 콜백"""
        state = self.LLMCallState(state_enum)
        result = result_ptr.contents

        if state == self.LLMCallState.RKLLM_RUN_NORMAL:
            if result.text:
                self.response_text += result.text.decode('utf-8', errors='ignore')
        return 0

    def _preprocess_image(self, image_data):
        """이미지 전처리 (numpy array 또는 base64)"""
        import cv2
        import numpy as np
        import base64

        # base64 문자열인 경우 디코딩
        if isinstance(image_data, str):
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = image_data

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 정사각형으로 확장
        h, w = img.shape[:2]
        size = max(h, w)
        square_img = np.full((size, size, 3), 127, dtype=np.uint8)
        x_off, y_off = (size - w) // 2, (size - h) // 2
        square_img[y_off:y_off+h, x_off:x_off+w] = img

        # 리사이즈
        resized = cv2.resize(square_img, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

        # 정규화 및 변환
        tensor = resized.astype(np.float32)
        tensor = (tensor / 255.0 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)   # Add batch

        return tensor.astype(np.float32)

    def inference(self, image_data, prompt, model_name=None):
        """
        VLM 추론 수행

        Args:
            image_data: numpy array (BGR) 또는 base64 문자열
            prompt: 사용자 프롬프트
            model_name: 모델 이름 (없으면 기본 모델 사용)

        Returns:
            str: 생성된 응답 텍스트
        """
        import ctypes
        import numpy as np

        if not self.is_ready:
            if not self.initialize(model_name):
                return "🖥️ RK3588 VLM 모델 초기화 실패"

        try:
            # 이미지 전처리
            input_tensor = self._preprocess_image(image_data)

            # Vision Encoder 실행
            img_vec_output = self.vision_session.run(
                [self.vision_output_name],
                {self.vision_input_name: input_tensor}
            )[0]
            img_vec = img_vec_output.flatten().astype(np.float32)

            # LLM 추론
            self.response_text = ""

            rkllm_input = self.RKLLMInput()
            rkllm_input.role = b"user"
            rkllm_input.input_type = self.RKLLMInputType.RKLLM_INPUT_MULTIMODAL

            full_prompt = f"Picture 1: <image> {prompt}"
            rkllm_input.multimodal_input.prompt = full_prompt.encode('utf-8')
            rkllm_input.multimodal_input.image_embed = img_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rkllm_input.multimodal_input.n_image_tokens = img_vec_output.shape[0]
            rkllm_input.multimodal_input.n_image = 1
            rkllm_input.multimodal_input.image_height = self.IMAGE_HEIGHT
            rkllm_input.multimodal_input.image_width = self.IMAGE_WIDTH

            infer_params = self.RKLLMInferParam()
            infer_params.mode = self.RKLLMInferMode.RKLLM_INFER_GENERATE
            infer_params.keep_history = 0

            self.rk_llm.run(rkllm_input, infer_params)

            return self.response_text.strip() if self.response_text else "No response"

        except Exception as e:
            print(f"[RK3588-VLM] 추론 오류: {e}")
            import traceback
            traceback.print_exc()
            return f"🖥️ RK3588 VLM 오류: {str(e)}"

# 글로벌 인스턴스 (지연 초기화)
_rk3588_vlm_instance = None

def get_rk3588_vlm():
    """RK3588 VLM 싱글톤 인스턴스 반환"""
    global _rk3588_vlm_instance
    if _rk3588_vlm_instance is None:
        _rk3588_vlm_instance = RK3588LocalVLM()
    return _rk3588_vlm_instance

# =============================================================================
# API 키 관리 (API Key Management)
# =============================================================================

def load_api_keys():
    """
    보안 파일(.dev.vars)에서 API 키를 로드합니다.

    이 함수는 다음 순서로 API 키를 찾습니다:
    1. 애플리케이션 디렉토리의 .dev.vars 파일
    2. 사용자 홈 디렉토리의 .dev.vars 파일
    3. 환경 변수 (폴백)

    Returns:
        dict: 각 프로바이더별 API 키 딕셔너리
              {"gemini": "...", "groq": "...", "claude": "...", "openai": "...", "xai": "..."}

    보안 참고:
    - .dev.vars 파일은 chmod 600으로 권한 설정 권장
    - 절대 git에 커밋하지 않도록 .gitignore에 추가
    """
    # API 키 저장용 딕셔너리 초기화
    keys = {
        "gemini": "",   # Google Gemini API
        "groq": "",     # Groq API (고속 추론)
        "claude": "",   # Anthropic Claude API
        "openai": "",   # OpenAI API (GPT-4o, Whisper, TTS)
        "xai": "",      # xAI Grok API
    }

    # .dev.vars 파일 검색 경로 (우선순위 순)
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '.dev.vars'),  # 스크립트 디렉토리
        os.path.expanduser('~/.dev.vars'),                      # 홈 디렉토리
        '/home/orangepi/dual_npu_demo/.dev.vars',               # 고정 경로
    ]

    # 파일에서 API 키 로드
    for dev_vars_path in possible_paths:
        if os.path.exists(dev_vars_path):
            try:
                with open(dev_vars_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # 빈 줄 또는 주석 건너뛰기
                        if not line or line.startswith('#'):
                            continue
                        # KEY=VALUE 형식 파싱
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # 환경 변수명을 내부 키 이름으로 매핑
                            if key in ['GEMINI_API_KEY', 'GOOGLE_API_KEY']:
                                keys['gemini'] = value
                            elif key == 'GROQ_API_KEY':
                                keys['groq'] = value
                            elif key in ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY']:
                                keys['claude'] = value
                            elif key == 'OPENAI_API_KEY':
                                keys['openai'] = value
                            elif key == 'XAI_API_KEY':
                                keys['xai'] = value
                print(f"[Config] Loaded API keys from: {dev_vars_path}")
                break  # 첫 번째로 찾은 파일만 사용
            except Exception as e:
                print(f"[Config] Error loading {dev_vars_path}: {e}")

    # 환경 변수에서 폴백 로드 (파일에 없는 키만)
    if not keys['gemini']:
        keys['gemini'] = os.environ.get("GEMINI_API_KEY", "")
    if not keys['groq']:
        keys['groq'] = os.environ.get("GROQ_API_KEY", "")
    if not keys['claude']:
        keys['claude'] = os.environ.get("ANTHROPIC_API_KEY", "")

    return keys

# API 키 전역 로드
API_KEYS = load_api_keys()


# =============================================================================
# UI 다국어 번역 시스템 (UI Translation System)
# =============================================================================
# 지원 언어: 한국어, 영어, 일본어, 중국어, 스페인어, 프랑스어, 독일어,
#           포르투갈어, 러시아어, 아랍어

# 지원 언어 목록 (표시명, 언어 코드)
UI_LANGUAGES = [
    ("한국어", "ko"),
    ("English", "en"),
    ("日本語", "ja"),
    ("中文", "zh"),
    ("Español", "es"),
    ("Français", "fr"),
    ("Deutsch", "de"),
    ("Português", "pt"),
    ("Русский", "ru"),
    ("العربية", "ar"),
]

UI_TRANSLATIONS = {
    "ko": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 통역",
        "settings_btn": "⚙️ 설정",
        "dx_panel_title": "DX-M1 NPU - {model} Object Detection",
        "llm_panel_title": "RK3588 NPU - LLM Chat",
        "system_monitor": "System Monitor",
        "fps": "FPS",
        "inf_time": "추론",
        "objects": "객체",
        "npu_status": "NPU",
        "detected": "감지됨",
        "none": "없음",
        "model": "모델",
        "input_placeholder": "메시지를 입력하세요...",
        "what_see": "뭐가 보여?",
        "analyze": "분석해줘",
        "explain": "설명해줘",
        "what_see_prompt": "이 화면에서 무엇이 보이나요?",
        "analyze_prompt": "이 화면을 자세히 분석해주세요",
        "explain_prompt": "이 장면을 설명해주세요",
        "auto_desc_prompt": "지금 보이는 화면을 간단히 설명해주세요. 3문장 이내로.",
        "auto": "⏱️ 자동",
        "llm_init": "LLM: 초기화 중...",
        "frame_captured": "프레임 캡처됨! 분석 중...",
        "no_frame": "프레임 없음",
        "api_key_missing": "API 키 없음!",
        "translation_mode_title": "🌐 실시간 통역 모드",
        "lang1": "언어 1:",
        "lang2": "언어 2:",
        "swap_lang": "⇅ 언어 교환",
        "trans_info": "언어1로 말하면 → 언어2로 번역\n언어2로 말하면 → 언어1로 번역",
        "auto_desc_settings": "자동 설명 설정",
        "interval_sec": "간격 (초):",
        "cpu": "CPU",
        "temp": "온도",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: 활성",
        "npu_error": "NPU: 오류",
        "ai_assistant": "🤖 AI Vision Assistant",
        "ai_hint": "카메라에 보이는 물체에 대해 질문하세요.",
        "ui_language": "UI 언어:",
        "settings_title": "설정",
    },
    "en": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Translate",
        "settings_btn": "⚙️ Settings",
        "dx_panel_title": "DX-M1 NPU - {model} Object Detection",
        "llm_panel_title": "RK3588 NPU - LLM Chat",
        "system_monitor": "System Monitor",
        "fps": "FPS",
        "inf_time": "Inf",
        "objects": "Objects",
        "npu_status": "NPU",
        "detected": "Detected",
        "none": "None",
        "model": "Model",
        "input_placeholder": "Enter your message...",
        "what_see": "What's this?",
        "analyze": "Analyze",
        "explain": "Explain",
        "what_see_prompt": "What do you see in this image?",
        "analyze_prompt": "Please analyze this image in detail",
        "explain_prompt": "Please explain this scene",
        "auto_desc_prompt": "Please briefly describe what you see on the screen. In 3 sentences or less.",
        "auto": "⏱️ Auto",
        "llm_init": "LLM: Initializing...",
        "frame_captured": "Frame captured! Analyzing...",
        "no_frame": "No frame available",
        "api_key_missing": "API Key Missing!",
        "translation_mode_title": "🌐 Real-time Translation Mode",
        "lang1": "Language 1:",
        "lang2": "Language 2:",
        "swap_lang": "⇅ Swap Languages",
        "trans_info": "Speak in Lang1 → Translated to Lang2\nSpeak in Lang2 → Translated to Lang1",
        "auto_desc_settings": "Auto Description Settings",
        "interval_sec": "Interval (sec):",
        "cpu": "CPU",
        "temp": "TEMP",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Active",
        "npu_error": "NPU: Error",
        "ai_assistant": "🤖 AI Vision Assistant",
        "ai_hint": "Ask about objects visible in the camera.",
        "ui_language": "UI Language:",
        "settings_title": "Settings",
    },
    "ja": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 通訳",
        "settings_btn": "⚙️ 設定",
        "dx_panel_title": "DX-M1 NPU - {model} 物体検出",
        "llm_panel_title": "RK3588 NPU - LLM チャット",
        "system_monitor": "システムモニター",
        "fps": "FPS",
        "inf_time": "推論",
        "objects": "オブジェクト",
        "npu_status": "NPU",
        "detected": "検出",
        "none": "なし",
        "model": "モデル",
        "input_placeholder": "メッセージを入力...",
        "what_see": "何が見える?",
        "analyze": "分析して",
        "explain": "説明して",
        "what_see_prompt": "この画像に何が見えますか？",
        "analyze_prompt": "この画像を詳しく分析してください",
        "explain_prompt": "このシーンを説明してください",
        "auto_desc_prompt": "今見えている画面を簡潔に説明してください。3文以内で。",
        "auto": "⏱️ 自動",
        "llm_init": "LLM: 初期化中...",
        "frame_captured": "フレームキャプチャ！分析中...",
        "no_frame": "フレームなし",
        "api_key_missing": "APIキーがありません！",
        "translation_mode_title": "🌐 リアルタイム通訳モード",
        "lang1": "言語 1:",
        "lang2": "言語 2:",
        "swap_lang": "⇅ 言語交換",
        "trans_info": "言語1で話す → 言語2に翻訳\n言語2で話す → 言語1に翻訳",
        "auto_desc_settings": "自動説明設定",
        "interval_sec": "間隔 (秒):",
        "cpu": "CPU",
        "temp": "温度",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: アクティブ",
        "npu_error": "NPU: エラー",
        "ai_assistant": "🤖 AI ビジョンアシスタント",
        "ai_hint": "カメラに映る物体について質問してください。",
        "ui_language": "UI言語:",
        "settings_title": "設定",
    },
    "zh": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 翻译",
        "settings_btn": "⚙️ 设置",
        "dx_panel_title": "DX-M1 NPU - {model} 目标检测",
        "llm_panel_title": "RK3588 NPU - LLM 聊天",
        "system_monitor": "系统监控",
        "fps": "帧率",
        "inf_time": "推理",
        "objects": "对象",
        "npu_status": "NPU",
        "detected": "检测到",
        "none": "无",
        "model": "模型",
        "input_placeholder": "输入消息...",
        "what_see": "看到什么?",
        "analyze": "分析",
        "explain": "解释",
        "what_see_prompt": "这张图片里你看到了什么？",
        "analyze_prompt": "请详细分析这张图片",
        "explain_prompt": "请解释这个场景",
        "auto_desc_prompt": "请简要描述您在屏幕上看到的内容。不超过3句话。",
        "auto": "⏱️ 自动",
        "llm_init": "LLM: 初始化中...",
        "frame_captured": "帧已捕获！分析中...",
        "no_frame": "无可用帧",
        "api_key_missing": "缺少API密钥！",
        "translation_mode_title": "🌐 实时翻译模式",
        "lang1": "语言 1:",
        "lang2": "语言 2:",
        "swap_lang": "⇅ 交换语言",
        "trans_info": "用语言1说话 → 翻译成语言2\n用语言2说话 → 翻译成语言1",
        "auto_desc_settings": "自动描述设置",
        "interval_sec": "间隔 (秒):",
        "cpu": "CPU",
        "temp": "温度",
        "ram": "内存",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: 活跃",
        "npu_error": "NPU: 错误",
        "ai_assistant": "🤖 AI 视觉助手",
        "ai_hint": "询问摄像头中可见的物体。",
        "ui_language": "界面语言:",
        "settings_title": "设置",
    },
    "es": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Traducir",
        "settings_btn": "⚙️ Ajustes",
        "dx_panel_title": "DX-M1 NPU - {model} Detección de Objetos",
        "llm_panel_title": "RK3588 NPU - Chat LLM",
        "system_monitor": "Monitor del Sistema",
        "fps": "FPS",
        "inf_time": "Inf",
        "objects": "Objetos",
        "npu_status": "NPU",
        "detected": "Detectado",
        "none": "Ninguno",
        "model": "Modelo",
        "input_placeholder": "Escribe tu mensaje...",
        "what_see": "¿Qué ves?",
        "analyze": "Analizar",
        "explain": "Explicar",
        "what_see_prompt": "¿Qué ves en esta imagen?",
        "analyze_prompt": "Por favor, analiza esta imagen en detalle",
        "explain_prompt": "Por favor, explica esta escena",
        "auto_desc_prompt": "Por favor, describe brevemente lo que ves en la pantalla. En 3 oraciones o menos.",
        "auto": "⏱️ Auto",
        "llm_init": "LLM: Inicializando...",
        "frame_captured": "¡Captura! Analizando...",
        "no_frame": "Sin imagen",
        "api_key_missing": "¡Falta API Key!",
        "translation_mode_title": "🌐 Modo Traducción en Tiempo Real",
        "lang1": "Idioma 1:",
        "lang2": "Idioma 2:",
        "swap_lang": "⇅ Intercambiar",
        "trans_info": "Habla en Idioma1 → Traduce a Idioma2\nHabla en Idioma2 → Traduce a Idioma1",
        "auto_desc_settings": "Config. Descripción Auto",
        "interval_sec": "Intervalo (seg):",
        "cpu": "CPU",
        "temp": "TEMP",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Activo",
        "npu_error": "NPU: Error",
        "ai_assistant": "🤖 Asistente de Visión AI",
        "ai_hint": "Pregunta sobre objetos visibles en la cámara.",
        "ui_language": "Idioma UI:",
        "settings_title": "Ajustes",
    },
    "fr": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Traduire",
        "settings_btn": "⚙️ Paramètres",
        "dx_panel_title": "DX-M1 NPU - {model} Détection d'Objets",
        "llm_panel_title": "RK3588 NPU - Chat LLM",
        "system_monitor": "Moniteur Système",
        "fps": "FPS",
        "inf_time": "Inf",
        "objects": "Objets",
        "npu_status": "NPU",
        "detected": "Détecté",
        "none": "Aucun",
        "model": "Modèle",
        "input_placeholder": "Entrez votre message...",
        "what_see": "Que vois-tu?",
        "analyze": "Analyser",
        "explain": "Expliquer",
        "what_see_prompt": "Que voyez-vous dans cette image?",
        "analyze_prompt": "Veuillez analyser cette image en détail",
        "explain_prompt": "Veuillez expliquer cette scène",
        "auto_desc_prompt": "Veuillez décrire brièvement ce que vous voyez à l'écran. En 3 phrases maximum.",
        "auto": "⏱️ Auto",
        "llm_init": "LLM: Initialisation...",
        "frame_captured": "Capture! Analyse...",
        "no_frame": "Pas d'image",
        "api_key_missing": "Clé API manquante!",
        "translation_mode_title": "🌐 Mode Traduction Temps Réel",
        "lang1": "Langue 1:",
        "lang2": "Langue 2:",
        "swap_lang": "⇅ Échanger",
        "trans_info": "Parlez en Langue1 → Traduit en Langue2\nParlez en Langue2 → Traduit en Langue1",
        "auto_desc_settings": "Config. Description Auto",
        "interval_sec": "Intervalle (sec):",
        "cpu": "CPU",
        "temp": "TEMP",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Actif",
        "npu_error": "NPU: Erreur",
        "ai_assistant": "🤖 Assistant Vision AI",
        "ai_hint": "Posez des questions sur les objets visibles.",
        "ui_language": "Langue UI:",
        "settings_title": "Paramètres",
    },
    "de": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Übersetzen",
        "settings_btn": "⚙️ Einstellungen",
        "dx_panel_title": "DX-M1 NPU - {model} Objekterkennung",
        "llm_panel_title": "RK3588 NPU - LLM Chat",
        "system_monitor": "Systemmonitor",
        "fps": "FPS",
        "inf_time": "Inf",
        "objects": "Objekte",
        "npu_status": "NPU",
        "detected": "Erkannt",
        "none": "Keine",
        "model": "Modell",
        "input_placeholder": "Nachricht eingeben...",
        "what_see": "Was siehst du?",
        "analyze": "Analysieren",
        "explain": "Erklären",
        "what_see_prompt": "Was sehen Sie in diesem Bild?",
        "analyze_prompt": "Bitte analysieren Sie dieses Bild im Detail",
        "explain_prompt": "Bitte erklären Sie diese Szene",
        "auto_desc_prompt": "Bitte beschreiben Sie kurz, was Sie auf dem Bildschirm sehen. In maximal 3 Sätzen.",
        "auto": "⏱️ Auto",
        "llm_init": "LLM: Initialisierung...",
        "frame_captured": "Erfasst! Analyse...",
        "no_frame": "Kein Bild",
        "api_key_missing": "API-Schlüssel fehlt!",
        "translation_mode_title": "🌐 Echtzeit-Übersetzungsmodus",
        "lang1": "Sprache 1:",
        "lang2": "Sprache 2:",
        "swap_lang": "⇅ Tauschen",
        "trans_info": "In Sprache1 sprechen → Übersetzt in Sprache2\nIn Sprache2 sprechen → Übersetzt in Sprache1",
        "auto_desc_settings": "Auto-Beschreibung Einst.",
        "interval_sec": "Intervall (Sek):",
        "cpu": "CPU",
        "temp": "TEMP",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Aktiv",
        "npu_error": "NPU: Fehler",
        "ai_assistant": "🤖 AI Vision Assistent",
        "ai_hint": "Fragen Sie nach sichtbaren Objekten.",
        "ui_language": "UI-Sprache:",
        "settings_title": "Einstellungen",
    },
    "pt": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Traduzir",
        "settings_btn": "⚙️ Configurações",
        "dx_panel_title": "DX-M1 NPU - {model} Detecção de Objetos",
        "llm_panel_title": "RK3588 NPU - Chat LLM",
        "system_monitor": "Monitor do Sistema",
        "fps": "FPS",
        "inf_time": "Inf",
        "objects": "Objetos",
        "npu_status": "NPU",
        "detected": "Detectado",
        "none": "Nenhum",
        "model": "Modelo",
        "input_placeholder": "Digite sua mensagem...",
        "what_see": "O que vê?",
        "analyze": "Analisar",
        "explain": "Explicar",
        "what_see_prompt": "O que você vê nesta imagem?",
        "analyze_prompt": "Por favor, analise esta imagem em detalhes",
        "explain_prompt": "Por favor, explique esta cena",
        "auto_desc_prompt": "Por favor, descreva brevemente o que você vê na tela. Em no máximo 3 frases.",
        "auto": "⏱️ Auto",
        "llm_init": "LLM: Inicializando...",
        "frame_captured": "Capturado! Analisando...",
        "no_frame": "Sem imagem",
        "api_key_missing": "Chave API ausente!",
        "translation_mode_title": "🌐 Modo Tradução em Tempo Real",
        "lang1": "Idioma 1:",
        "lang2": "Idioma 2:",
        "swap_lang": "⇅ Trocar",
        "trans_info": "Fale em Idioma1 → Traduz para Idioma2\nFale em Idioma2 → Traduz para Idioma1",
        "auto_desc_settings": "Config. Descrição Auto",
        "interval_sec": "Intervalo (seg):",
        "cpu": "CPU",
        "temp": "TEMP",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Ativo",
        "npu_error": "NPU: Erro",
        "ai_assistant": "🤖 Assistente de Visão AI",
        "ai_hint": "Pergunte sobre objetos visíveis na câmera.",
        "ui_language": "Idioma UI:",
        "settings_title": "Configurações",
    },
    "ru": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 Перевод",
        "settings_btn": "⚙️ Настройки",
        "dx_panel_title": "DX-M1 NPU - {model} Детекция Объектов",
        "llm_panel_title": "RK3588 NPU - LLM Чат",
        "system_monitor": "Системный Монитор",
        "fps": "FPS",
        "inf_time": "Инф",
        "objects": "Объекты",
        "npu_status": "NPU",
        "detected": "Обнаружено",
        "none": "Нет",
        "model": "Модель",
        "input_placeholder": "Введите сообщение...",
        "what_see": "Что видишь?",
        "analyze": "Анализ",
        "explain": "Объясни",
        "what_see_prompt": "Что вы видите на этом изображении?",
        "analyze_prompt": "Пожалуйста, проанализируйте это изображение подробно",
        "explain_prompt": "Пожалуйста, объясните эту сцену",
        "auto_desc_prompt": "Пожалуйста, кратко опишите то, что вы видите на экране. Не более 3 предложений.",
        "auto": "⏱️ Авто",
        "llm_init": "LLM: Инициализация...",
        "frame_captured": "Захвачено! Анализ...",
        "no_frame": "Нет кадра",
        "api_key_missing": "API ключ отсутствует!",
        "translation_mode_title": "🌐 Режим Перевода в Реальном Времени",
        "lang1": "Язык 1:",
        "lang2": "Язык 2:",
        "swap_lang": "⇅ Поменять",
        "trans_info": "Говорите на Языке1 → Перевод на Язык2\nГоворите на Языке2 → Перевод на Язык1",
        "auto_desc_settings": "Настр. Авто-описания",
        "interval_sec": "Интервал (сек):",
        "cpu": "CPU",
        "temp": "ТЕМП",
        "ram": "RAM",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: Активен",
        "npu_error": "NPU: Ошибка",
        "ai_assistant": "🤖 AI Визуальный Ассистент",
        "ai_hint": "Спросите о видимых объектах.",
        "ui_language": "Язык интерфейса:",
        "settings_title": "Настройки",
    },
    "ar": {
        "app_title": "Production Dual NPU Demo",
        "translate_btn": "🌐 ترجمة",
        "settings_btn": "⚙️ إعدادات",
        "dx_panel_title": "DX-M1 NPU - {model} كشف الكائنات",
        "llm_panel_title": "RK3588 NPU - دردشة LLM",
        "system_monitor": "مراقب النظام",
        "fps": "FPS",
        "inf_time": "استدلال",
        "objects": "كائنات",
        "npu_status": "NPU",
        "detected": "مكتشف",
        "none": "لا شيء",
        "model": "نموذج",
        "input_placeholder": "أدخل رسالتك...",
        "what_see": "ماذا ترى؟",
        "analyze": "تحليل",
        "explain": "شرح",
        "what_see_prompt": "ماذا ترى في هذه الصورة؟",
        "analyze_prompt": "من فضلك حلل هذه الصورة بالتفصيل",
        "explain_prompt": "من فضلك اشرح هذا المشهد",
        "auto_desc_prompt": "من فضلك صف بإيجاز ما تراه على الشاشة. في 3 جمل أو أقل.",
        "auto": "⏱️ تلقائي",
        "llm_init": "LLM: جاري التهيئة...",
        "frame_captured": "تم الالتقاط! جاري التحليل...",
        "no_frame": "لا يوجد إطار",
        "api_key_missing": "مفتاح API مفقود!",
        "translation_mode_title": "🌐 وضع الترجمة الفورية",
        "lang1": "اللغة 1:",
        "lang2": "اللغة 2:",
        "swap_lang": "⇅ تبديل",
        "trans_info": "تحدث بـاللغة1 ← ترجم إلى اللغة2\nتحدث بـاللغة2 ← ترجم إلى اللغة1",
        "auto_desc_settings": "إعدادات الوصف التلقائي",
        "interval_sec": "الفاصل (ثانية):",
        "cpu": "CPU",
        "temp": "حرارة",
        "ram": "ذاكرة",
        "gpu": "GPU",
        "rk_npu": "RK-NPU",
        "dx_npu": "DX-NPU",
        "npu_active": "NPU: نشط",
        "npu_error": "NPU: خطأ",
        "ai_assistant": "🤖 مساعد الرؤية AI",
        "ai_hint": "اسأل عن الكائنات المرئية في الكاميرا.",
        "ui_language": "لغة الواجهة:",
        "settings_title": "إعدادات",
    },
}

# Current UI language (default: Korean)
# 현재 UI 언어 설정 (기본값: 한국어)
current_ui_lang = "ko"


def get_text(key, **kwargs):
    """
    현재 설정된 UI 언어에 맞는 번역 텍스트를 반환합니다.

    Args:
        key (str): 번역 키 (예: "settings_btn", "what_see")
        **kwargs: 포맷 문자열에 전달할 인자 (예: model="YOLOX-S")

    Returns:
        str: 번역된 텍스트. 키가 없으면 키 자체를 반환

    Example:
        >>> get_text("dx_panel_title", model="YOLOX-S")
        "DX-M1 NPU - YOLOX-S Object Detection"
    """
    text = UI_TRANSLATIONS.get(current_ui_lang, UI_TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# =============================================================================
# Vision LLM 클라이언트 (Vision LLM Client)
# =============================================================================

class VisionLLMClient:
    """
    다양한 클라우드 Vision LLM API를 호출하는 클라이언트 클래스

    지원 프로바이더:
    - Google Gemini (gemini-2.0-flash, gemini-1.5-pro)
    - Groq (LLaVA - 고속 추론)
    - Anthropic Claude (claude-sonnet-4, claude-3-haiku)
    - OpenAI (gpt-4o, gpt-4o-mini)
    - xAI Grok (grok-vision-beta)

    모든 메서드는 정적 메서드로 구현되어 있어 인스턴스 생성 없이 사용 가능
    """

    @staticmethod
    def encode_image(frame):
        """
        OpenCV 프레임을 Base64 인코딩된 JPEG 문자열로 변환

        Args:
            frame (np.ndarray): OpenCV BGR 이미지 프레임

        Returns:
            str: Base64 인코딩된 JPEG 이미지 문자열

        Note:
            JPEG 품질은 85%로 설정 (파일 크기와 품질의 균형)
        """
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def call_gemini(image_base64, prompt, api_key):
        """Call Google Gemini Vision API"""
        if not api_key:
            return "Gemini API key not set. Set GEMINI_API_KEY environment variable."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")
            else:
                return f"Gemini Error: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    @staticmethod
    def call_groq(image_base64, prompt, api_key):
        """Call Groq Vision API (LLaVA)"""
        if not api_key:
            return "Groq API key not set. Set GROQ_API_KEY environment variable."

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llava-v1.5-7b-4096-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                return f"Groq Error: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"Groq Error: {str(e)}"

    @staticmethod
    def call_claude(image_base64, prompt, api_key, model="claude-sonnet-4-20250514"):
        """Call Claude Vision API"""
        if not api_key:
            return "Claude API key not set. Set ANTHROPIC_API_KEY environment variable."

        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("content", [{}])[0].get("text", "No response")
            else:
                return f"Claude Error: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"Claude Error: {str(e)}"

    @staticmethod
    def call_openai(image_base64, prompt, api_key, model="gpt-4o"):
        """Call OpenAI Vision API (GPT-4o, GPT-4o-mini)"""
        if not api_key:
            return "OpenAI API key not set. Set OPENAI_API_KEY environment variable."

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                return f"OpenAI Error: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    @staticmethod
    def call_xai(image_base64, prompt, api_key, model="grok-vision-beta"):
        """Call X.AI Grok Vision API"""
        if not api_key:
            return "X.AI API key not set. Set XAI_API_KEY environment variable."

        url = "https://api.x.ai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                return f"Grok Error: {response.status_code} - {response.text[:200]}"
        except Exception as e:
            return f"Grok Error: {str(e)}"

    @staticmethod
    def call_local_vlm(image_base64, prompt, model="qwen2.5-vl-3b"):
        """
        RK3588 로컬 VLM 추론 (직접 NPU 사용)

        Args:
            image_base64: Base64 인코딩된 이미지
            prompt: 사용자 프롬프트
            model: 로컬 VLM 모델 ID (기본: qwen2.5-vl-3b)

        Returns:
            str: 생성된 응답 텍스트

        지원 모델 (NPU 직접 추론):
            - qwen2.5-vl-3b: 3B 파라미터, Vision 4초, LLM 8 tokens/s
            - 기타 모델은 HTTP API 서버 필요 (RKLLM 1.2.1 호환 모델만)
        """
        # NPU 직접 추론 지원 모델 (RKLLM 1.2.1 호환)
        DIRECT_NPU_MODELS = ["qwen2.5-vl-3b"]

        if model in DIRECT_NPU_MODELS:
            try:
                import sys
                sys.stdout.flush()
                print(f"[RK3588-VLM] 직접 추론 시작 - model: {model}", flush=True)
                vlm = get_rk3588_vlm()
                print(f"[RK3588-VLM] VLM 인스턴스 획득, is_ready: {vlm.is_ready}, current: {vlm.current_model}", flush=True)
                result = vlm.inference(image_base64, prompt, model)
                print(f"[RK3588-VLM] 추론 완료, 응답 길이: {len(result) if result else 0}", flush=True)
                return result
            except Exception as e:
                import traceback
                print(f"[RK3588-VLM] 직접 추론 실패: {e}", flush=True)
                traceback.print_exc()
                # 실패 시 오류 메시지 직접 반환 (HTTP 폴백 안함)
                return f"🖥️ RK3588 VLM 오류: {str(e)}"

        # 기타 모델 또는 폴백: HTTP API 서버 사용
        url = f"{LOCAL_VLM_API_URL}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 512,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                return f"Local VLM Error: {response.status_code} - {error_msg}"
        except requests.exceptions.ConnectionError:
            return (
                "🖥️ RK3588 VLM 모델을 로드할 수 없습니다.\n\n"
                "모델 경로: /mnt/external/rkllm_models/Qwen2.5-VL-3B/\n"
                "필요 파일: vision_encoder.rknn, language_model_w8a8.rkllm"
            )
        except Exception as e:
            return f"Local VLM Error: {str(e)}"

# ===== STT/TTS Classes =====

class SpeechToText(QThread):
    """OpenAI Whisper-based Speech-to-Text"""
    transcription_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    recording_state = pyqtSignal(bool)  # True=recording, False=stopped

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Whisper optimal sample rate

    def start_recording(self):
        """Start recording audio"""
        if not AUDIO_AVAILABLE:
            self.status_changed.emit("Audio not available")
            return

        self.recording = True
        self.audio_data = []
        self.recording_state.emit(True)
        self.status_changed.emit("Recording...")
        self.start()

    def stop_recording(self):
        """Stop recording and transcribe"""
        self.recording = False
        self.recording_state.emit(False)

    def run(self):
        """Record audio until stopped"""
        try:
            def callback(indata, frames, time, status):
                if self.recording:
                    self.audio_data.append(indata.copy())

            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                              dtype='float32', callback=callback):
                while self.recording:
                    sd.sleep(100)

            # Process recorded audio
            if self.audio_data:
                self.status_changed.emit("Transcribing...")
                audio = np.concatenate(self.audio_data, axis=0)
                text = self.transcribe(audio)
                if text:
                    self.transcription_ready.emit(text)
                    self.status_changed.emit("Ready")
                else:
                    self.status_changed.emit("No speech detected")
            else:
                self.status_changed.emit("No audio recorded")

        except Exception as e:
            self.status_changed.emit(f"Recording error: {str(e)[:30]}")
            print(f"[STT] Error: {e}")

    def transcribe(self, audio_data):
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_data, self.sample_rate)
                temp_path = f.name

            # Call OpenAI Whisper API (auto-detect language with multilingual prompt)
            client = OpenAI(api_key=self.api_key)
            with open(temp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    # Multilingual prompt helps with language detection
                    prompt="This audio may be in English, Korean, Japanese, or Chinese."
                )

            # Cleanup
            import os
            os.unlink(temp_path)

            return transcript.text.strip()

        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None


class TextToSpeech(QThread):
    """OpenAI TTS-based Text-to-Speech"""
    playback_started = pyqtSignal()
    playback_finished = pyqtSignal()
    status_changed = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.text_queue = []
        self.lock = threading.Lock()
        self.running = True
        self.voice = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer
        self.speed = 1.0

    def set_voice(self, voice):
        """Set TTS voice"""
        self.voice = voice

    def speak(self, text):
        """Add text to speech queue"""
        with self.lock:
            self.text_queue.append(text)

    def run(self):
        """Process speech queue"""
        while self.running:
            text = None
            with self.lock:
                if self.text_queue:
                    text = self.text_queue.pop(0)

            if text:
                self.generate_and_play(text)
            else:
                time.sleep(0.1)

    def generate_and_play(self, text):
        """Generate speech and play it"""
        if not AUDIO_AVAILABLE:
            return

        try:
            self.status_changed.emit("Generating speech...")
            self.playback_started.emit()

            # Call OpenAI TTS API
            client = OpenAI(api_key=self.api_key)

            # Save to temp file and play (using with_streaming_response to fix deprecation bug)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name

            with client.audio.speech.with_streaming_response.create(
                model="tts-1-hd",  # High quality model
                voice=self.voice,
                input=text,
                speed=self.speed
            ) as response:
                response.stream_to_file(temp_path)

            # Play audio
            self.status_changed.emit("Speaking...")
            data, samplerate = sf.read(temp_path)
            # Resample to 48000Hz for HDMI output
            target_sr = 48000
            if samplerate != target_sr:
                from scipy import signal
                num_samples = int(len(data) * target_sr / samplerate)
                data = signal.resample(data, num_samples)
            # Convert mono to stereo for HDMI output
            if len(data.shape) == 1:
                data = np.column_stack([data, data])
            sd.play(data, target_sr)
            sd.wait()  # Wait until playback is done

            # Cleanup
            import os
            os.unlink(temp_path)

            self.playback_finished.emit()
            self.status_changed.emit("Ready")

        except Exception as e:
            print(f"[TTS] Error: {e}")
            self.status_changed.emit(f"TTS error: {str(e)[:30]}")
            self.playback_finished.emit()

    def stop(self):
        self.running = False
        sd.stop()


class AutoDescriptionWorker(QThread):
    """Auto-describe scene at intervals"""
    description_request = pyqtSignal()  # Request to describe current scene
    status_changed = pyqtSignal(str)

    def __init__(self, interval_seconds=30):
        super().__init__()
        self.interval = interval_seconds
        self.running = True
        self.enabled = False

    def set_interval(self, seconds):
        self.interval = max(5, seconds)  # Minimum 5 seconds

    def set_enabled(self, enabled):
        self.enabled = enabled

    def run(self):
        elapsed = 0
        while self.running:
            if self.enabled:
                if elapsed >= self.interval:
                    print(f"[AutoDesc Worker] Emitting description_request (interval={self.interval}s)")
                    self.description_request.emit()
                    elapsed = 0
                elapsed += 1
            else:
                elapsed = 0
            time.sleep(1)

    def stop(self):
        self.running = False


SUPPORTED_LANGUAGES = [
    # East Asian
    ("한국어", "Korean"),
    ("中文 (简体)", "Chinese Simplified"),
    ("中文 (繁體)", "Chinese Traditional"),
    ("日本語", "Japanese"),
    # European
    ("English", "English"),
    ("Español", "Spanish"),
    ("Français", "French"),
    ("Deutsch", "German"),
    ("Italiano", "Italian"),
    ("Português", "Portuguese"),
    ("Nederlands", "Dutch"),
    ("Polski", "Polish"),
    ("Русский", "Russian"),
    ("Українська", "Ukrainian"),
    ("Ελληνικά", "Greek"),
    ("Čeština", "Czech"),
    ("Svenska", "Swedish"),
    ("Norsk", "Norwegian"),
    ("Dansk", "Danish"),
    ("Suomi", "Finnish"),
    ("Magyar", "Hungarian"),
    ("Română", "Romanian"),
    ("Български", "Bulgarian"),
    ("Hrvatski", "Croatian"),
    ("Slovenčina", "Slovak"),
    # Middle Eastern
    ("العربية", "Arabic"),
    ("עברית", "Hebrew"),
    ("فارسی", "Persian"),
    ("Türkçe", "Turkish"),
    # South Asian
    ("हिंदी", "Hindi"),
    ("বাংলা", "Bengali"),
    ("தமிழ்", "Tamil"),
    ("తెలుగు", "Telugu"),
    ("मराठी", "Marathi"),
    ("ગુજરાતી", "Gujarati"),
    ("ಕನ್ನಡ", "Kannada"),
    ("മലയാളം", "Malayalam"),
    ("ਪੰਜਾਬੀ", "Punjabi"),
    ("اردو", "Urdu"),
    # Southeast Asian
    ("ไทย", "Thai"),
    ("Tiếng Việt", "Vietnamese"),
    ("Bahasa Indonesia", "Indonesian"),
    ("Bahasa Melayu", "Malay"),
    ("Filipino", "Filipino"),
    ("မြန်မာ", "Burmese"),
    ("ខ្មែរ", "Khmer"),
    ("ລາວ", "Lao"),
    # African
    ("Kiswahili", "Swahili"),
    ("Afrikaans", "Afrikaans"),
]


class TranslationSettingsDialog(QDialog):
    """Settings dialog for translation/interpreter mode"""

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("🌐 Translation Mode Settings")
        self.setModal(True)
        self.setFixedSize(350, 280)
        self.setStyleSheet("""
            QDialog { background: #1a1a2e; }
            QLabel { color: #c9d1d9; }
            QComboBox { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 8px; min-width: 150px; }
            QComboBox:drop-down { border: none; }
            QComboBox QAbstractItemView { background: #21262d; color: #c9d1d9; selection-background-color: #238636; }
            QCheckBox { color: #c9d1d9; }
            QPushButton { background: #238636; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background: #2ea043; }
        """)

        if current_settings is None:
            current_settings = {'enabled': True, 'lang1': 'Korean', 'lang2': 'English'}

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("🌐 실시간 통역 모드")
        title.setStyleSheet("color: #58a6ff; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Enable checkbox - default to True when opening dialog
        self.enable_check = QCheckBox("통역 모드 활성화")
        self.enable_check.setChecked(True)  # Default ON
        layout.addWidget(self.enable_check)

        # Language 1
        lang1_layout = QHBoxLayout()
        lang1_label = QLabel("언어 1:")
        lang1_label.setFixedWidth(60)
        lang1_layout.addWidget(lang1_label)
        self.lang1_combo = QComboBox()
        for display, code in SUPPORTED_LANGUAGES:
            self.lang1_combo.addItem(display, code)
        idx = next((i for i, (d, c) in enumerate(SUPPORTED_LANGUAGES) if c == current_settings.get('lang1', 'Korean')), 0)
        self.lang1_combo.setCurrentIndex(idx)
        lang1_layout.addWidget(self.lang1_combo)
        layout.addLayout(lang1_layout)

        # Swap button
        swap_btn = QPushButton("⇅ 언어 교환")
        swap_btn.setStyleSheet("background: #30363d; padding: 6px;")
        swap_btn.clicked.connect(self.swap_languages)
        layout.addWidget(swap_btn)

        # Language 2
        lang2_layout = QHBoxLayout()
        lang2_label = QLabel("언어 2:")
        lang2_label.setFixedWidth(60)
        lang2_layout.addWidget(lang2_label)
        self.lang2_combo = QComboBox()
        for display, code in SUPPORTED_LANGUAGES:
            self.lang2_combo.addItem(display, code)
        idx = next((i for i, (d, c) in enumerate(SUPPORTED_LANGUAGES) if c == current_settings.get('lang2', 'English')), 1)
        self.lang2_combo.setCurrentIndex(idx)
        lang2_layout.addWidget(self.lang2_combo)
        layout.addLayout(lang2_layout)

        # Info
        info = QLabel("언어1로 말하면 → 언어2로 번역\n언어2로 말하면 → 언어1로 번역")
        info.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(info)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def swap_languages(self):
        idx1 = self.lang1_combo.currentIndex()
        idx2 = self.lang2_combo.currentIndex()
        self.lang1_combo.setCurrentIndex(idx2)
        self.lang2_combo.setCurrentIndex(idx1)

    def get_settings(self):
        return {
            'enabled': self.enable_check.isChecked(),
            'lang1': self.lang1_combo.currentData(),
            'lang2': self.lang2_combo.currentData(),
            'lang1_display': self.lang1_combo.currentText(),
            'lang2_display': self.lang2_combo.currentText(),
        }


class AutoDescriptionSettingsDialog(QDialog):
    """Settings dialog for auto-description feature and UI language"""

    def __init__(self, parent=None, current_interval=30, current_enabled=False, current_ui_language="ko"):
        super().__init__(parent)
        self.setWindowTitle(get_text("settings_title"))
        self.setModal(True)
        self.setFixedSize(350, 280)
        self.setStyleSheet("""
            QDialog { background: #1a1a2e; }
            QLabel { color: #c9d1d9; }
            QSpinBox { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 5px; }
            QCheckBox { color: #c9d1d9; }
            QComboBox { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 5px; min-width: 150px; }
            QComboBox:drop-down { border: none; }
            QComboBox QAbstractItemView { background: #21262d; color: #c9d1d9; selection-background-color: #238636; }
            QPushButton { background: #238636; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background: #2ea043; }
        """)

        layout = QVBoxLayout(self)

        # UI Language selector
        lang_layout = QHBoxLayout()
        lang_label = QLabel(get_text("ui_language"))
        lang_layout.addWidget(lang_label)

        self.lang_combo = QComboBox()
        current_lang_idx = 0
        for i, (display, code) in enumerate(UI_LANGUAGES):
            self.lang_combo.addItem(display, code)
            if code == current_ui_language:
                current_lang_idx = i
        self.lang_combo.setCurrentIndex(current_lang_idx)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #30363d;")
        layout.addWidget(line)

        # Auto description section label
        auto_desc_label = QLabel(get_text("auto_desc_settings"))
        auto_desc_label.setStyleSheet("color: #58a6ff; font-weight: bold; margin-top: 8px;")
        layout.addWidget(auto_desc_label)

        # Enable checkbox
        self.enable_check = QCheckBox("Enable Auto Description")
        self.enable_check.setChecked(current_enabled)
        layout.addWidget(self.enable_check)

        # Interval setting
        interval_layout = QHBoxLayout()
        interval_label = QLabel(get_text("interval_sec"))
        interval_layout.addWidget(interval_label)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(5, 300)
        self.interval_spin.setValue(current_interval)
        self.interval_spin.setSuffix(" sec")
        interval_layout.addWidget(self.interval_spin)
        layout.addLayout(interval_layout)

        # Voice output checkbox
        self.voice_output_check = QCheckBox("Speak description aloud")
        self.voice_output_check.setChecked(True)
        layout.addWidget(self.voice_output_check)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        return {
            'enabled': self.enable_check.isChecked(),
            'interval': self.interval_spin.value(),
            'voice_output': self.voice_output_check.isChecked(),
            'ui_language': self.lang_combo.currentData()
        }


# YOLO Model Selection - YOLOX-S is recommended (Apache 2.0 license, no PPU)
# Available models:
#   - YOLOv5s_640.dxnn: Original, proven stable (PPU)
#   - YOLOv9-S-2.dxnn: Higher accuracy (mAP50: 46.7%), ~106 FPS (PPU)
#   - YOLOXS-1.dxnn: Apache 2.0 license, 405 FPS, anchor-free (non-PPU)
YOLO_MODEL = "YOLOX-S"  # Options: "YOLOv5s", "YOLOv9-S", "YOLOX-S"

MODEL_PATHS = {
    "YOLOv5s": "/home/orangepi/model_for_demo/YOLOv5s_640.dxnn",
    "YOLOv9-S": "/home/orangepi/model_for_demo/YOLOv9-S-2.dxnn",
    "YOLOX-S": "/home/orangepi/model_for_demo/YOLOXS-1.dxnn",
}
MODEL_PATH = MODEL_PATHS.get(YOLO_MODEL, MODEL_PATHS["YOLOX-S"])

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)


def boxes_overlap_or_close(boxA, boxB):
    """
    Check if two boxes should be merged using DYNAMIC thresholds.
    center_dist_thresh = 20% of larger box's diagonal (not hardcoded)
    """
    # Calculate box dimensions
    wa, ha = boxA[2] - boxA[0], boxA[3] - boxA[1]
    wb, hb = boxB[2] - boxB[0], boxB[3] - boxB[1]

    # Dynamic center distance threshold: 20% of larger box diagonal
    # Reduced to allow multiple people detection
    diag_a = (wa**2 + ha**2)**0.5
    diag_b = (wb**2 + hb**2)**0.5
    center_dist_thresh = max(diag_a, diag_b) * 0.2  # 20% of diagonal (was 50%)

    # Check IoU - higher threshold for multi-person
    iou = calculate_iou(boxA, boxB)
    if iou > 0.3:  # 30% overlap = same object (was 5%)
        return True

    # Get centers
    cx_a, cy_a = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
    cx_b, cy_b = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2

    # Check center distance (dynamic threshold)
    dist = ((cx_a - cx_b)**2 + (cy_a - cy_b)**2)**0.5
    if dist < center_dist_thresh:
        return True

    # Check if either center is inside the other box
    if boxA[0] <= cx_b <= boxA[2] and boxA[1] <= cy_b <= boxA[3]:
        return True
    if boxB[0] <= cx_a <= boxB[2] and boxB[1] <= cy_a <= boxB[3]:
        return True

    return False


def weighted_box_fusion(detections):
    """
    Merge overlapping boxes using dynamic thresholds.
    Uses boxes_overlap_or_close() which has dynamic center_dist based on box diagonal.
    """
    if not detections:
        return []

    # Sort by confidence (highest first)
    dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    merged_results = []

    while dets:
        base_det = dets.pop(0)
        group = [base_det]

        # Find boxes that should be merged (same class, overlapping/close)
        remaining_dets = []
        for other in dets:
            if other['class_id'] == base_det['class_id']:
                if boxes_overlap_or_close(base_det['box'], other['box']):
                    group.append(other)
                else:
                    remaining_dets.append(other)
            else:
                remaining_dets.append(other)
        dets = remaining_dets

        # Use the highest confidence box directly (no averaging distortion)
        # base_det already has the highest score in the group
        merged_results.append({
            'box': base_det['box'],
            'score': base_det['score'],
            'class': base_det['class'],
            'class_id': base_det['class_id']
        })

    return merged_results


class ProIOUTracker:
    """
    IOU-based tracker with Stability Filter for photo detection.
    - Uses IoU instead of centroid distance for matching
    - Tracks W/H ratio variance to detect static images (photos)
    """

    def __init__(self, max_disappeared=30, iou_threshold=0.45):
        self.next_id = 0
        self.objects = {}  # ID -> {'box', 'class', 'score', 'history'}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _register(self, det):
        w = det['box'][2] - det['box'][0]
        h = det['box'][3] - det['box'][1]
        ratio = w / max(h, 1)
        self.objects[self.next_id] = {
            'box': det['box'],
            'class': det['class'],
            'class_id': det['class_id'],
            'score': det['score'],
            'history': deque([ratio], maxlen=30)
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        # Reset ID counter when all objects are gone to prevent ID jumping
        if not self.objects:
            self.next_id = 0

    def update(self, detections):
        """Update tracker - returns all tracked objects"""
        objects, _ = self.update_with_matches(detections)
        return objects

    def update_with_matches(self, detections):
        """Update tracker and return (all_objects, matched_ids_this_frame)"""
        matched_this_frame = set()

        # Handle no detections
        if not detections:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects, matched_this_frame

        # If no existing objects, register all detections
        if not self.objects:
            for det in detections:
                self._register(det)
                matched_this_frame.add(self.next_id - 1)
            return self.objects, matched_this_frame

        # IOU-based matching
        obj_ids = list(self.objects.keys())
        prev_boxes = [self.objects[oid]['box'] for oid in obj_ids]
        current_dets = list(detections)
        matched_obj_ids = set()
        matched_det_indices = set()

        # Build IOU matrix and match greedily
        for i, oid in enumerate(obj_ids):
            best_iou = 0
            best_det_idx = -1
            for j, det in enumerate(current_dets):
                if j in matched_det_indices:
                    continue
                iou = calculate_iou(prev_boxes[i], det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = j

            if best_iou > self.iou_threshold and best_det_idx >= 0:
                # Update existing object
                det = current_dets[best_det_idx]
                w = det['box'][2] - det['box'][0]
                h = det['box'][3] - det['box'][1]
                ratio = w / max(h, 1)

                # Less smoothing for more responsive tracking
                old_box = self.objects[oid]['box']
                alpha = 0.85  # Higher = more responsive to new detection
                new_box = tuple(int(alpha * det['box'][k] + (1-alpha) * old_box[k]) for k in range(4))

                self.objects[oid]['box'] = new_box
                self.objects[oid]['score'] = det['score']
                self.objects[oid]['history'].append(ratio)
                self.disappeared[oid] = 0
                matched_obj_ids.add(oid)
                matched_det_indices.add(best_det_idx)
                matched_this_frame.add(oid)

        # Mark unmatched objects as disappeared
        for oid in obj_ids:
            if oid not in matched_obj_ids:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        # Register new detections
        for j, det in enumerate(current_dets):
            if j not in matched_det_indices:
                self._register(det)
                matched_this_frame.add(self.next_id - 1)

        return self.objects, matched_this_frame

    def is_alive(self, obj_id):
        """
        Stability Filter: Check if object is a real moving person or a static photo.
        Photos have near-zero variance in W/H ratio over time.
        Also checks box area - very small boxes are likely noise.
        """
        if obj_id not in self.objects:
            return False

        obj = self.objects[obj_id]
        box = obj['box']

        # Check minimum area - reject tiny detections
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        if area < 2000:  # Minimum 2000 pixels (roughly 45x45)
            return False

        history = obj['history']
        if len(history) < 60:  # Need 60 samples (~3 sec at 20fps)
            return True

        variance = np.var(list(history))
        # Photo: variance = 0 (perfectly static)
        # Real person: variance > 0 (even minimal breathing/movement)
        # Threshold 0.0003: low enough to not filter still people, but catches photos
        return variance > 0.0003


# =============================================================================
# DX-M1 NPU 객체 검출기 (DX-M1 NPU Object Detector)
# =============================================================================

class DXM1Detector:
    """
    DeepX DX-M1 NPU를 사용한 실시간 객체 검출 클래스

    이 클래스는 DeepX DX-M1 NPU 가속기를 활용하여 YOLO 기반의
    객체 검출을 수행합니다. PPU(Post Processing Unit) 모델과
    비-PPU 모델 모두 지원합니다.

    지원 모델:
    - PPU 모델 (YOLOv5s, YOLOv9-S):
        출력: DeviceBoundingBox_t 구조체 (32바이트/객체)
        후처리가 하드웨어에서 수행되어 빠름

    - 비-PPU 모델 (YOLOX-S):
        출력: Raw 텐서 [1, 8400, 85]
        [x, y, w, h, obj_conf, 80개 클래스 점수]
        소프트웨어에서 NMS 수행 필요

    주요 기능:
    - 실시간 객체 검출 (최대 ~400 FPS YOLOX-S)
    - 다중 객체 추적 (ProIOUTracker 사용)
    - 최대 10명까지 동시 인식
    - 정적 이미지(사진) 필터링

    Attributes:
        model_path (str): DXNN 모델 파일 경로
        model_size (int): 입력 이미지 크기 (640x640)
        conf_threshold (float): 객체 신뢰도 임계값
        score_threshold (float): 최종 점수 임계값
        nms_threshold (float): NMS IoU 임계값
        max_persons (int): 최대 인식 인원수
    """

    def __init__(self, model_path):
        """
        DXM1Detector 초기화

        Args:
            model_path (str): DXNN 모델 파일 경로
                             예: "/home/orangepi/model_for_demo/YOLOXS-1.dxnn"
        """
        self.model_path = model_path
        self.ie = None                          # DeepX InferenceEngine 인스턴스
        self.model_size = 640                   # 모든 모델 640x640 사용
        self.initialized = False                # 초기화 완료 여부
        self.is_ppu = False                     # PPU 모델 여부 (초기화 시 설정)
        self.conf_threshold = 0.25              # 객체 신뢰도 임계값
        self.score_threshold = 0.55             # 균형잡힌 최종 점수 임계값
        self.nms_threshold = 0.4                # NMS IoU 임계값

        # 객체 추적기 설정
        self.tracker = ProIOUTracker(
            max_disappeared=90,                 # 90프레임(~4.5초) 후 객체 삭제
            iou_threshold=0.15                  # 추적 IoU 임계값
        )
        self.max_persons = 10                   # 최대 인식 인원수

        # 검출 실패 시 위치 예측용
        self.last_valid_detection = None        # 마지막 유효 검출 결과
        self.frames_without_detection = 0       # 검출 실패 연속 프레임 수

        # 입력 형식 정보 (초기화 시 설정)
        self.input_shape = None

    def initialize(self):
        try:
            from dx_engine import InferenceEngine
            self.ie = InferenceEngine(self.model_path)
            self.initialized = True

            # Get model info
            input_info = self.ie.get_input_tensors_info()
            output_info = self.ie.get_output_tensors_info()
            self.is_ppu = self.ie.is_ppu()

            # Store input shape for preprocessing
            if input_info:
                self.input_shape = input_info[0]['shape']

            print(f"[DX-M1] Model loaded: {self.model_path}")
            print(f"[DX-M1] Input: {input_info}")
            print(f"[DX-M1] Output: {output_info}")
            print(f"[DX-M1] Is PPU: {self.is_ppu}")
            print(f"[DX-M1] Input shape: {self.input_shape}")
            return True
        except Exception as e:
            print(f"[DX-M1] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """Letterbox resize maintaining aspect ratio"""
        shape = img.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Symmetric padding
        top = bottom = int(dh)
        left = right = int(dw)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)

        # Return actual left/top padding used (not halved values)
        return img, r, (float(left), float(top))

    def detect(self, frame):
        """Run detection and return results with tracking"""
        if not self.initialized or self.ie is None:
            return [], frame

        try:
            orig_h, orig_w = frame.shape[:2]

            # Letterbox preprocessing
            img, ratio, (dw, dh) = self.letterbox(frame, self.model_size)

            # Prepare input based on model type
            if self.is_ppu:
                # PPU models (YOLOv5s, YOLOv9-S): [1, 640, 1920] flattened row-wise
                input_data = img.reshape(1, self.model_size, self.model_size * 3).astype(np.uint8)
            else:
                # Non-PPU models (YOLOX-S): [1, 640, 640, 3] NHWC format
                input_data = img.reshape(1, self.model_size, self.model_size, 3).astype(np.uint8)

            # Run inference
            outputs = self.ie.run([input_data.flatten()])

            # Parse detections based on model type
            if self.is_ppu:
                raw_detections = self.parse_detections_ppu(outputs[0], orig_w, orig_h, ratio, dw, dh)
            else:
                raw_detections = self.parse_detections_yolox(outputs[0], orig_w, orig_h, ratio, dw, dh)

            # STEP 1: Weighted Box Fusion - merge overlapping ghost boxes
            fused_detections = weighted_box_fusion(raw_detections)

            # STEP 2: Keep top N detections per class (person class)
            # max_persons controls how many people can be detected
            final_detections = self.keep_top_per_class(fused_detections, max_per_class=self.max_persons)

            # STEP 3: IOU-based tracking for stable IDs
            tracked_objects, matched_ids = self.tracker.update_with_matches(final_detections)

            # STEP 4: Only return CURRENTLY MATCHED objects (not old tracked ones)
            detections = []
            for obj_id in matched_ids:
                if obj_id not in tracked_objects:
                    continue
                obj_data = tracked_objects[obj_id]

                # Apply Stability Filter: reject photos (near-zero variance)
                if not self.tracker.is_alive(obj_id):
                    continue

                detections.append({
                    'id': obj_id,
                    'class': obj_data['class'],
                    'class_id': obj_data['class_id'],
                    'score': obj_data['score'],
                    'box': obj_data['box']
                })

            # FINAL SAFEGUARD: Keep only top N detections (prevents multiple person boxes)
            if len(detections) > self.max_persons:
                detections = sorted(detections, key=lambda x: x['score'], reverse=True)[:self.max_persons]

            # Force stable ID when single-person mode (prevents ID jumping)
            if self.max_persons == 1 and detections:
                detections[0]['id'] = 0

            # Position prediction: if no detections but had recent valid detection
            if detections:
                self.last_valid_detection = detections[0].copy()
                self.frames_without_detection = 0
            else:
                self.frames_without_detection += 1
                # Use last known position for up to 15 frames (~0.75 sec at 20fps)
                if self.last_valid_detection and self.frames_without_detection <= 15:
                    predicted = self.last_valid_detection.copy()
                    # Reduce score to indicate prediction
                    predicted['score'] = max(0.3, predicted['score'] - 0.1 * self.frames_without_detection)
                    detections = [predicted]

            # Draw on frame with tracked boxes
            result_frame = self.draw_boxes(frame, detections)

            return detections, result_frame

        except Exception as e:
            print(f"[DX-M1] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return [], frame

    def parse_detections_yolox(self, output, orig_w, orig_h, ratio, dw, dh):
        """
        Parse YOLOX-S raw tensor output (non-PPU) with grid-based decoding.

        Output shape: [1, 8400, 85]
        8400 = 80*80 + 40*40 + 20*20 (grids for strides 8, 16, 32)
        Each detection: [x_offset, y_offset, w_exp, h_exp, obj_conf, cls_0, ..., cls_79]

        Decoding:
        - x_center = (x_offset + grid_x) * stride
        - y_center = (y_offset + grid_y) * stride
        - width = exp(w_exp) * stride
        - height = exp(h_exp) * stride
        """
        detections = []

        # Reshape output to [8400, 85]
        try:
            output = output.reshape(-1, 85)
        except Exception as e:
            print(f"[YOLOX] Output reshape error: {e}, shape: {output.shape}")
            return detections

        # Generate grids for YOLOX anchor-free decoding
        # YOLOX-S uses strides [8, 16, 32] with feature maps [80x80, 40x40, 20x20]
        grids = []
        strides = []
        for stride in [8, 16, 32]:
            grid_size = self.model_size // stride
            yv, xv = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
            grid = np.stack([xv.flatten(), yv.flatten()], axis=1)
            grids.append(grid)
            strides.append(np.full((grid_size * grid_size, 1), stride))

        grids = np.concatenate(grids, axis=0)  # [8400, 2]
        strides = np.concatenate(strides, axis=0)  # [8400, 1]

        num_boxes = output.shape[0]

        for i in range(num_boxes):
            data = output[i]

            # Extract objectness confidence (already sigmoid)
            obj_conf = data[4]
            if obj_conf < self.conf_threshold:
                continue

            # Get class scores (already sigmoid) and find best class
            class_scores = data[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]

            # Final score = objectness * class_score
            score = obj_conf * class_score
            if score < self.score_threshold:
                continue

            # Filter invalid labels
            if class_id >= len(COCO_CLASSES):
                continue

            # Decode coordinates using grid and stride
            grid_x, grid_y = grids[i]
            stride = strides[i, 0]

            # YOLOX decoding formula
            cx = (data[0] + grid_x) * stride
            cy = (data[1] + grid_y) * stride
            w = np.exp(data[2]) * stride
            h = np.exp(data[3]) * stride

            # Filter out abnormally large boxes (> 85% of model size)
            if w > self.model_size * 0.85 and h > self.model_size * 0.85:
                continue

            # Convert from center to corner format
            x1_model = cx - w / 2
            y1_model = cy - h / 2
            x2_model = cx + w / 2
            y2_model = cy + h / 2

            # Remove letterbox padding and scale back to original
            x1 = (x1_model - dw) / ratio
            y1 = (y1_model - dh) / ratio
            x2 = (x2_model - dw) / ratio
            y2 = (y2_model - dh) / ratio

            # Clamp to image bounds
            x1 = max(0, min(orig_w, int(x1)))
            y1 = max(0, min(orig_h, int(y1)))
            x2 = max(0, min(orig_w, int(x2)))
            y2 = max(0, min(orig_h, int(y2)))

            # Valid box check
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = COCO_CLASSES[class_id]

            detections.append({
                'class': class_name,
                'class_id': int(class_id),
                'score': float(score),
                'box': (x1, y1, x2, y2)
            })

        # Apply NMS per class
        detections = self.apply_nms(detections)
        return detections

    def parse_detections_ppu(self, output, orig_w, orig_h, ratio, dw, dh):
        """
        Parse DeviceBoundingBox_t array from PPU output (YOLOv5s, YOLOv9-S)

        Struct layout (32 bytes):
        - float x (4)      : center x NORMALIZED (0-1)
        - float y (4)      : center y NORMALIZED (0-1)
        - float w (4)      : width NORMALIZED (0-1)
        - float h (4)      : height NORMALIZED (0-1)
        - uint8 grid_y (1) : grid position y
        - uint8 grid_x (1) : grid position x
        - uint8 box_idx (1): box index
        - uint8 layer_idx(1): layer index
        - float score (4)  : confidence score
        - uint32 label (4) : class label (0-79 for COCO)
        - padding (4)      : unused
        """
        detections = []

        if output.nbytes == 0:
            return detections

        # Each detection is 32 bytes
        num_dets = output.nbytes // 32
        raw = output.tobytes()

        for i in range(num_dets):
            # Unpack: 4 floats + 4 bytes + 1 float + 1 uint32 + 4 bytes padding
            # '<4f4B1fI4x' = little-endian: 4 floats, 4 uint8, 1 float, 1 uint32, 4 padding
            data = struct.unpack('<4f4BfI4x', raw[i*32:(i+1)*32])

            # Coordinates are NORMALIZED (0-1), multiply by model size (640)
            cx = data[0] * self.model_size
            cy = data[1] * self.model_size
            w = data[2] * self.model_size
            h = data[3] * self.model_size

            grid_y, grid_x, box_idx, layer_idx = data[4], data[5], data[6], data[7]
            score = data[8]
            label = data[9]

            # Filter by score
            if score < self.score_threshold:
                continue

            # Filter only truly full-screen boxes (both dimensions > 85%)
            if data[2] > 0.85 and data[3] > 0.85:
                continue

            # Filter invalid labels
            if label >= len(COCO_CLASSES):
                continue

            # Convert from center to corner format (in model space - pixels)
            x1_model = cx - w / 2
            y1_model = cy - h / 2
            x2_model = cx + w / 2
            y2_model = cy + h / 2

            # Remove letterbox padding and scale back to original
            x1 = (x1_model - dw) / ratio
            y1 = (y1_model - dh) / ratio
            x2 = (x2_model - dw) / ratio
            y2 = (y2_model - dh) / ratio

            # Clamp to image bounds
            x1 = max(0, min(orig_w, int(x1)))
            y1 = max(0, min(orig_h, int(y1)))
            x2 = max(0, min(orig_w, int(x2)))
            y2 = max(0, min(orig_h, int(y2)))

            # Valid box check
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = COCO_CLASSES[label]

            detections.append({
                'class': class_name,
                'class_id': label,
                'score': float(score),
                'box': (x1, y1, x2, y2)
            })

        # Apply NMS per class
        detections = self.apply_nms(detections)
        return detections

    def keep_top_per_class(self, detections, max_per_class=1):
        """
        Keep only the top N highest-scoring detections per class.
        For webcam demo: max_per_class=1 means only 1 person box, eliminating all ghosts.
        """
        if not detections:
            return []

        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class_id']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)

        # Keep top N per class
        result = []
        for cls, dets in by_class.items():
            sorted_dets = sorted(dets, key=lambda x: x['score'], reverse=True)
            result.extend(sorted_dets[:max_per_class])

        return result

    def spatial_nms(self, detections, grid_size=150):
        """
        Keep only the top detection per class, plus check spatial proximity.
        For webcam use case, typically only 1-2 people in frame.
        """
        if len(detections) <= 1:
            return detections

        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class_id']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)

        kept = []
        for cls, dets in by_class.items():
            # Sort by score
            sorted_dets = sorted(dets, key=lambda x: x['score'], reverse=True)

            # Keep top detection
            if sorted_dets:
                top = sorted_dets[0]
                kept.append(top)

                # Only keep additional detections if they're far from existing ones
                top_cx = (top['box'][0] + top['box'][2]) / 2
                top_cy = (top['box'][1] + top['box'][3]) / 2

                for det in sorted_dets[1:]:
                    cx = (det['box'][0] + det['box'][2]) / 2
                    cy = (det['box'][1] + det['box'][3]) / 2
                    dist = ((cx - top_cx)**2 + (cy - top_cy)**2)**0.5

                    # Only keep if very far apart (>300 pixels = different person)
                    if dist > 300:
                        kept.append(det)
                        break  # Max 2 per class

        return kept

    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression per class"""
        if len(detections) == 0:
            return detections

        # Group by class
        class_groups = {}
        for det in detections:
            cls = det['class_id']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(det)

        # Apply NMS per class
        result = []
        for cls, dets in class_groups.items():
            if len(dets) == 1:
                result.extend(dets)
                continue

            # Prepare for cv2.dnn.NMSBoxes
            boxes = []
            scores = []
            for d in dets:
                x1, y1, x2, y2 = d['box']
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                scores.append(d['score'])

            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_threshold)
            if len(indices) > 0:
                indices = indices.flatten()
                for i in indices:
                    result.append(dets[i])

        return result

    def draw_boxes(self, frame, detections):
        """Draw bounding boxes with tracking IDs"""
        result = frame.copy()

        # Distinct bright colors for each person ID (max 10 people)
        PERSON_COLORS = [
            (0, 255, 0),      # ID 0: Green
            (255, 0, 0),      # ID 1: Blue
            (0, 0, 255),      # ID 2: Red
            (255, 255, 0),    # ID 3: Cyan
            (255, 0, 255),    # ID 4: Magenta
            (0, 255, 255),    # ID 5: Yellow
            (255, 128, 0),    # ID 6: Orange
            (128, 0, 255),    # ID 7: Purple
            (0, 255, 128),    # ID 8: Spring Green
            (255, 128, 128),  # ID 9: Light Coral
        ]

        for det in detections:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            label = det['class']
            obj_id = det.get('id', 0)

            # Use tracking ID for consistent distinct color
            color = PERSON_COLORS[obj_id % len(PERSON_COLORS)]

            # Draw thick box for tracked objects
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

            # Draw label with ID
            text = f"#{obj_id} {label} {score:.0%}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)

            # Draw label text (white on colored background)
            cv2.putText(result, text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result


# =============================================================================
# 제스처 인식 클래스 (Gesture Recognition Classes)
# =============================================================================

# COCO 포즈 키포인트 인덱스
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 스켈레톤 연결 정보 (시각화용)
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11),  # 왼쪽 다리
    (16, 14), (14, 12),  # 오른쪽 다리
    (11, 12),            # 엉덩이
    (5, 11), (6, 12),    # 몸통-엉덩이
    (5, 6),              # 어깨
    (5, 7), (7, 9),      # 왼쪽 팔
    (6, 8), (8, 10),     # 오른쪽 팔
    (1, 2),              # 눈
    (0, 1), (0, 2),      # 코-눈
    (1, 3), (2, 4),      # 눈-귀
    (3, 5), (4, 6),      # 귀-어깨
]

# 스켈레톤 색상 (BGR)
SKELETON_COLORS = [
    (255, 153, 51),   # 다리 - 주황
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 51, 255),   # 엉덩이 - 핑크
    (255, 51, 255),
    (255, 51, 255),
    (0, 128, 255),    # 어깨 - 파랑
    (0, 128, 255),    # 왼팔
    (0, 128, 255),
    (0, 128, 255),    # 오른팔
    (0, 128, 255),
    (0, 255, 0),      # 얼굴 - 녹색
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
]


class GestureRecognizer:
    """
    키포인트 기반 제스처 인식 클래스

    17개 COCO 키포인트를 분석하여 다양한 제스처를 인식합니다.

    지원 제스처:
    - hands_up: 양손 들기
    - left_hand_up: 왼손 들기
    - right_hand_up: 오른손 들기
    - waving: 손 흔들기
    - pointing_left: 왼쪽 가리키기
    - pointing_right: 오른쪽 가리키기
    - arms_crossed: 팔짱 끼기
    - t_pose: T 포즈
    """

    def __init__(self):
        """제스처 인식기 초기화"""
        self.gesture_history = deque(maxlen=10)  # 제스처 히스토리 (안정화용)
        self.wrist_history = deque(maxlen=15)    # 손목 위치 히스토리 (흔들기 감지용)
        self.last_gesture = None
        self.gesture_confidence = 0.0

    def recognize(self, keypoints, confidence_threshold=0.5):
        """
        키포인트에서 제스처 인식

        Args:
            keypoints: 17개 키포인트 [(x, y, conf), ...]
            confidence_threshold: 최소 신뢰도 임계값

        Returns:
            dict: {
                'gesture': 제스처 이름,
                'confidence': 신뢰도,
                'details': 추가 정보
            }
        """
        if keypoints is None or len(keypoints) < 17:
            return {'gesture': 'unknown', 'confidence': 0.0, 'details': {}}

        # 키포인트 추출 (신뢰도 체크)
        kpts = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                if conf >= confidence_threshold:
                    kpts[name] = (x, y, conf)
                else:
                    kpts[name] = None
            else:
                kpts[name] = None

        # 손목 위치 히스토리 업데이트 (흔들기 감지용)
        if kpts.get('left_wrist') and kpts.get('right_wrist'):
            self.wrist_history.append({
                'left': kpts['left_wrist'][:2],
                'right': kpts['right_wrist'][:2],
                'time': time.time()
            })

        # 제스처 감지 (우선순위 순)
        gesture_result = self._detect_gestures(kpts)

        # 제스처 히스토리 업데이트 (안정화)
        self.gesture_history.append(gesture_result['gesture'])

        # 가장 빈번한 제스처 반환 (안정화)
        if len(self.gesture_history) >= 3:
            gesture_counts = Counter(self.gesture_history)
            most_common = gesture_counts.most_common(1)[0]
            if most_common[1] >= 2:  # 최소 2번 이상 감지
                gesture_result['gesture'] = most_common[0]
                gesture_result['confidence'] = most_common[1] / len(self.gesture_history)

        self.last_gesture = gesture_result
        return gesture_result

    def _detect_gestures(self, kpts):
        """제스처 감지 로직"""
        result = {'gesture': 'standing', 'confidence': 0.5, 'details': {}}

        # 필수 키포인트 체크
        has_shoulders = kpts.get('left_shoulder') and kpts.get('right_shoulder')
        has_wrists = kpts.get('left_wrist') and kpts.get('right_wrist')
        has_elbows = kpts.get('left_elbow') and kpts.get('right_elbow')

        if not has_shoulders:
            return result

        shoulder_y = (kpts['left_shoulder'][1] + kpts['right_shoulder'][1]) / 2
        shoulder_x_left = kpts['left_shoulder'][0]
        shoulder_x_right = kpts['right_shoulder'][0]
        shoulder_width = abs(shoulder_x_right - shoulder_x_left)

        # 1. 양손 들기 (Hands Up)
        if has_wrists:
            left_wrist_y = kpts['left_wrist'][1]
            right_wrist_y = kpts['right_wrist'][1]
            left_wrist_x = kpts['left_wrist'][0]
            right_wrist_x = kpts['right_wrist'][0]

            # 양손이 어깨보다 위에 있으면 "손 들기"
            left_up = left_wrist_y < shoulder_y - shoulder_width * 0.3
            right_up = right_wrist_y < shoulder_y - shoulder_width * 0.3

            if left_up and right_up:
                result = {
                    'gesture': 'hands_up',
                    'confidence': 0.9,
                    'details': {'both_hands': True}
                }
                return result
            elif left_up:
                result = {
                    'gesture': 'left_hand_up',
                    'confidence': 0.85,
                    'details': {'hand': 'left'}
                }
                return result
            elif right_up:
                result = {
                    'gesture': 'right_hand_up',
                    'confidence': 0.85,
                    'details': {'hand': 'right'}
                }
                return result

            # 2. 손 흔들기 (Waving) - 손목 움직임 분석
            if len(self.wrist_history) >= 8:
                waving = self._detect_waving()
                if waving['is_waving']:
                    result = {
                        'gesture': 'waving',
                        'confidence': waving['confidence'],
                        'details': {'hand': waving['hand']}
                    }
                    return result

            # 3. T 포즈 (T-Pose) - 양팔을 수평으로 벌림
            if has_elbows:
                # 팔꿈치와 손목이 어깨와 비슷한 높이
                elbow_y_left = kpts['left_elbow'][1]
                elbow_y_right = kpts['right_elbow'][1]

                arms_horizontal = (
                    abs(left_wrist_y - shoulder_y) < shoulder_width * 0.4 and
                    abs(right_wrist_y - shoulder_y) < shoulder_width * 0.4 and
                    abs(elbow_y_left - shoulder_y) < shoulder_width * 0.4 and
                    abs(elbow_y_right - shoulder_y) < shoulder_width * 0.4
                )

                # 손목이 어깨보다 바깥에 있어야 함
                arms_spread = (
                    left_wrist_x < shoulder_x_left - shoulder_width * 0.5 and
                    right_wrist_x > shoulder_x_right + shoulder_width * 0.5
                )

                if arms_horizontal and arms_spread:
                    result = {
                        'gesture': 't_pose',
                        'confidence': 0.9,
                        'details': {}
                    }
                    return result

            # 4. 포인팅 (Pointing)
            if has_elbows:
                # 한 팔이 뻗어있고 다른 팔은 내려있음
                left_arm_extended = (
                    abs(kpts['left_elbow'][1] - left_wrist_y) < shoulder_width * 0.3 and
                    left_wrist_x < shoulder_x_left - shoulder_width * 0.5
                )
                right_arm_extended = (
                    abs(kpts['right_elbow'][1] - right_wrist_y) < shoulder_width * 0.3 and
                    right_wrist_x > shoulder_x_right + shoulder_width * 0.5
                )

                if left_arm_extended and not right_arm_extended:
                    result = {
                        'gesture': 'pointing_left',
                        'confidence': 0.8,
                        'details': {'direction': 'left'}
                    }
                    return result
                elif right_arm_extended and not left_arm_extended:
                    result = {
                        'gesture': 'pointing_right',
                        'confidence': 0.8,
                        'details': {'direction': 'right'}
                    }
                    return result

            # 5. 팔짱 끼기 (Arms Crossed)
            if has_elbows:
                # 손목이 반대편 어깨 근처에 있음
                arms_crossed = (
                    abs(left_wrist_x - shoulder_x_right) < shoulder_width * 0.5 and
                    abs(right_wrist_x - shoulder_x_left) < shoulder_width * 0.5 and
                    left_wrist_y > shoulder_y and
                    right_wrist_y > shoulder_y
                )

                if arms_crossed:
                    result = {
                        'gesture': 'arms_crossed',
                        'confidence': 0.85,
                        'details': {}
                    }
                    return result

        return result

    def _detect_waving(self):
        """손목 움직임으로 흔들기 감지"""
        if len(self.wrist_history) < 8:
            return {'is_waving': False, 'confidence': 0.0, 'hand': None}

        # 최근 8개 프레임의 손목 위치 분석
        recent = list(self.wrist_history)[-8:]

        for hand in ['left', 'right']:
            x_positions = [h[hand][0] for h in recent if h.get(hand)]
            if len(x_positions) < 6:
                continue

            # 방향 변화 카운트 (좌우 흔들기)
            direction_changes = 0
            for i in range(1, len(x_positions)):
                if i > 0:
                    prev_dir = x_positions[i-1] - x_positions[max(0, i-2)] if i > 1 else 0
                    curr_dir = x_positions[i] - x_positions[i-1]
                    if prev_dir * curr_dir < 0:  # 방향 변화
                        direction_changes += 1

            # 3번 이상 방향 변화 = 흔들기
            if direction_changes >= 3:
                # 움직임 범위 체크
                x_range = max(x_positions) - min(x_positions)
                if x_range > 30:  # 최소 30픽셀 이동
                    return {
                        'is_waving': True,
                        'confidence': min(0.9, 0.6 + direction_changes * 0.1),
                        'hand': hand
                    }

        return {'is_waving': False, 'confidence': 0.0, 'hand': None}


class DXM1PoseDetector:
    """
    DeepX DX-M1 NPU를 사용한 포즈 추정 클래스

    YOLOv5Pose 모델을 사용하여 사람의 관절 위치(키포인트)를 감지합니다.
    17개 COCO 키포인트를 출력하며, 제스처 인식에 활용됩니다.

    Attributes:
        model_path (str): YOLOv5Pose DXNN 모델 파일 경로
        model_size (int): 입력 이미지 크기 (640x640)
        conf_threshold (float): 신뢰도 임계값
    """

    def __init__(self, model_path="/home/orangepi/model_for_demo/YOLOv5Pose640_1.dxnn"):
        """
        포즈 감지기 초기화

        Args:
            model_path: YOLOv5Pose 모델 파일 경로
        """
        self.model_path = model_path
        self.ie = None
        self.model_size = 640
        self.initialized = False
        self.conf_threshold = 0.5
        self.score_threshold = 0.5

        # 제스처 인식기
        self.gesture_recognizer = GestureRecognizer()

        # 앵커 설정 (YOLOv5Pose 640)
        self.anchors = {
            'layer0': {'grid': 80, 'anchors': [(10, 13), (16, 30), (33, 23)]},
            'layer1': {'grid': 40, 'anchors': [(30, 61), (62, 45), (59, 119)]},
            'layer2': {'grid': 20, 'anchors': [(116, 90), (156, 198), (373, 326)]},
        }

    def initialize(self):
        """NPU 초기화 및 모델 로드"""
        try:
            from dx_engine import InferenceEngine
            self.ie = InferenceEngine(self.model_path)
            self.initialized = True

            input_info = self.ie.get_input_tensors_info()
            output_info = self.ie.get_output_tensors_info()
            self.is_ppu = self.ie.is_ppu()

            print(f"[DX-M1 Pose] Model loaded: {self.model_path}")
            print(f"[DX-M1 Pose] Is PPU: {self.is_ppu}")
            print(f"[DX-M1 Pose] Output info: {output_info}")
            return True
        except Exception as e:
            print(f"[DX-M1 Pose] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        """Letterbox 전처리 (비율 유지 리사이즈)"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = bottom = int(dh)
        left = right = int(dw)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)
        return img, r, (float(left), float(top))

    def detect(self, frame):
        """
        포즈 감지 수행

        Args:
            frame: 입력 이미지 (BGR)

        Returns:
            tuple: (poses, result_frame, gestures)
                - poses: 감지된 포즈 목록 [{keypoints, box, score}, ...]
                - result_frame: 시각화된 프레임
                - gestures: 감지된 제스처 목록
        """
        if not self.initialized or self.ie is None:
            return [], frame, []

        try:
            orig_h, orig_w = frame.shape[:2]

            # 전처리
            img, ratio, (dw, dh) = self.letterbox(frame, self.model_size)

            # PPU 모델 입력 형식
            if self.is_ppu:
                input_data = img.reshape(1, self.model_size, self.model_size * 3).astype(np.uint8)
            else:
                input_data = img.reshape(1, self.model_size, self.model_size, 3).astype(np.uint8)

            # 추론 실행
            outputs = self.ie.run([input_data.flatten()])

            # 포즈 파싱
            poses = self._parse_poses(outputs[0], orig_w, orig_h, ratio, dw, dh)

            # 제스처 인식
            gestures = []
            for pose in poses:
                gesture = self.gesture_recognizer.recognize(pose['keypoints'])
                gestures.append(gesture)

            # 시각화
            result_frame = self.draw_poses(frame.copy(), poses, gestures)

            return poses, result_frame, gestures

        except Exception as e:
            print(f"[DX-M1 Pose] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return [], frame, []

    def _parse_poses(self, output, orig_w, orig_h, ratio, dw, dh):
        """포즈 출력 파싱"""
        poses = []

        try:
            # PPU 모델은 DevicePose_t 구조체 출력
            # 구조: [x, y, w, h, score, layer_idx, grid_x, grid_y, box_idx, kpts[17][3]]
            # 총 크기: 4 + 1 + 4 + 51 = 60 floats per detection

            data = np.frombuffer(output, dtype=np.float32)

            # PPU 모델 출력 파싱
            if self.is_ppu:
                # DevicePose_t 구조체 (60 floats)
                struct_size = 60
                num_detections = len(data) // struct_size

                for i in range(min(num_detections, 10)):  # 최대 10명
                    offset = i * struct_size

                    # 바운딩 박스 및 점수
                    x = data[offset + 0]
                    y = data[offset + 1]
                    w = data[offset + 2]
                    h = data[offset + 3]
                    score = data[offset + 4]

                    if score < self.score_threshold:
                        continue

                    # 좌표 변환 (letterbox 역변환)
                    x1 = (x - w/2 - dw) / ratio
                    y1 = (y - h/2 - dh) / ratio
                    x2 = (x + w/2 - dw) / ratio
                    y2 = (y + h/2 - dh) / ratio

                    # 범위 제한
                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))

                    # 키포인트 (17개, 각 3값: x, y, conf)
                    keypoints = []
                    kpt_offset = offset + 9  # 9 = 5 (box+score) + 4 (layer, grid, box indices)

                    for k in range(17):
                        kx = data[kpt_offset + k*3 + 0]
                        ky = data[kpt_offset + k*3 + 1]
                        kconf = data[kpt_offset + k*3 + 2]

                        # 좌표 변환
                        kx = (kx - dw) / ratio
                        ky = (ky - dh) / ratio
                        kx = max(0, min(orig_w, kx))
                        ky = max(0, min(orig_h, ky))

                        keypoints.append((kx, ky, kconf))

                    poses.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'score': float(score),
                        'keypoints': keypoints
                    })
            else:
                # Non-PPU 모델 (raw tensor 출력)
                # 이 경우 수동으로 후처리 필요
                # TODO: Non-PPU 모델 지원 추가
                pass

        except Exception as e:
            print(f"[DX-M1 Pose] Parse error: {e}")
            import traceback
            traceback.print_exc()

        return poses

    def draw_poses(self, frame, poses, gestures=None):
        """
        포즈 시각화

        Args:
            frame: 입력 프레임
            poses: 감지된 포즈 목록
            gestures: 감지된 제스처 목록

        Returns:
            시각화된 프레임
        """
        result = frame.copy()

        for idx, pose in enumerate(poses):
            keypoints = pose['keypoints']
            box = pose['box']
            score = pose['score']

            # 바운딩 박스 색상
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]

            # 바운딩 박스 그리기
            x1, y1, x2, y2 = box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # 스켈레톤 그리기
            points = []
            for kx, ky, kconf in keypoints:
                if kconf > self.conf_threshold:
                    points.append((int(kx), int(ky)))
                else:
                    points.append(None)

            # 연결선 그리기
            for i, (start, end) in enumerate(SKELETON_CONNECTIONS):
                if start < len(points) and end < len(points):
                    pt1 = points[start]
                    pt2 = points[end]
                    if pt1 and pt2:
                        skel_color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
                        cv2.line(result, pt1, pt2, skel_color, 2, cv2.LINE_AA)

            # 키포인트 그리기
            for i, pt in enumerate(points):
                if pt:
                    # 키포인트 색상 (부위별)
                    if i < 5:  # 얼굴
                        kpt_color = (0, 255, 0)
                    elif i < 11:  # 상체
                        kpt_color = (0, 128, 255)
                    else:  # 하체
                        kpt_color = (255, 153, 51)
                    cv2.circle(result, pt, 4, kpt_color, -1)
                    cv2.circle(result, pt, 5, (255, 255, 255), 1)

            # 제스처 표시
            if gestures and idx < len(gestures):
                gesture = gestures[idx]
                gesture_text = gesture['gesture'].replace('_', ' ').title()
                gesture_conf = gesture['confidence']

                # 제스처 라벨
                label = f"{gesture_text} ({gesture_conf:.0%})"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                # 배경
                cv2.rectangle(result, (x1, y1 - th - 15), (x1 + tw + 10, y1 - 5), color, -1)
                cv2.putText(result, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 점수 표시
            score_text = f"{score:.0%}"
            cv2.putText(result, score_text, (x2 - 50, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result


class DXM1FaceDetector:
    """
    OpenCV DNN 기반 얼굴 검출 클래스

    YuNet 얼굴 검출 모델 사용 (BSD 라이선스, 상업적 사용 가능)
    CPU에서 실행되며 빠른 실시간 검출 지원

    Attributes:
        conf_threshold (float): 신뢰도 임계값
        nms_threshold (float): NMS 임계값
    """

    def __init__(self, model_path=None):
        """
        얼굴 검출기 초기화
        """
        self.detector = None
        self.initialized = False
        self.conf_threshold = 0.6
        self.nms_threshold = 0.3
        self.input_size = (320, 320)

    def initialize(self):
        """OpenCV Haar Cascade 또는 DNN 기반 얼굴 검출기 초기화"""
        try:
            # OpenCV Haar Cascade 사용 (가장 호환성 좋음)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)

            if self.detector.empty():
                print("[Face] Failed to load Haar Cascade")
                return False

            self.initialized = True
            print(f"[Face] OpenCV Haar Cascade face detector initialized")
            return True

        except Exception as e:
            print(f"[Face] Init failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect(self, frame):
        """
        얼굴 검출 수행

        Args:
            frame: 입력 이미지 (BGR)

        Returns:
            tuple: (faces, result_frame)
        """
        if not self.initialized or self.detector is None:
            return [], frame

        try:
            result = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 얼굴 검출
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            faces = []
            for (x, y, w, h) in detections:
                face = {
                    'box': [int(x), int(y), int(x + w), int(y + h)],
                    'score': 0.99,  # Haar cascade doesn't provide score
                    'landmarks': None,
                    'emotion': None
                }
                faces.append(face)

            # 시각화
            result = self.visualize(result, faces)

            return faces, result

        except Exception as e:
            print(f"[Face] Detection error: {e}")
            return [], frame

    def visualize(self, frame, faces):
        """검출 결과 시각화"""
        result = frame.copy()

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['box']
            score = face['score']

            # 얼굴별 색상
            colors = [
                (255, 128, 0),    # 주황
                (0, 255, 128),    # 연두
                (128, 0, 255),    # 보라
                (255, 0, 128),    # 분홍
                (0, 128, 255),    # 하늘
            ]
            color = colors[i % len(colors)]

            # 바운딩 박스 (두꺼운 선)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

            # 라벨
            label = f"Face {i+1}"
            if face.get('emotion'):
                label += f" ({face['emotion']})"

            # 라벨 배경
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result

    def analyze_emotion_vlm(self, frame, face_box):
        """
        VLM을 사용하여 얼굴 감정 분석

        Args:
            frame: 원본 이미지 (BGR)
            face_box: [x1, y1, x2, y2] 얼굴 영역

        Returns:
            str: 감정 레이블 (happy, sad, angry, surprised, neutral, fear, disgust)
        """
        try:
            x1, y1, x2, y2 = face_box

            # 얼굴 영역에 여유 공간 추가 (더 정확한 분석을 위해)
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # 얼굴 크롭
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None

            # Base64 인코딩
            _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            face_base64 = base64.b64encode(buffer).decode('utf-8')

            # VLM 프롬프트
            prompt = """Analyze this person's facial expression and emotion.
Reply with ONLY ONE word from: happy, sad, angry, surprised, neutral, fear, disgust
Just the emotion word, nothing else."""

            # VLM 호출
            result = VisionLLMClient.call_local_vlm(face_base64, prompt, "qwen2.5-vl-3b")

            if result:
                # 응답에서 감정 추출
                result_lower = result.lower().strip()
                emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fear', 'disgust']
                for emotion in emotions:
                    if emotion in result_lower:
                        return emotion.capitalize()
                # 매칭되지 않으면 첫 단어 반환
                first_word = result_lower.split()[0] if result_lower else 'neutral'
                return first_word.capitalize()[:10]

            return None

        except Exception as e:
            print(f"[Face] Emotion analysis error: {e}")
            return None


class EmotionAnalyzerThread(QThread):
    """
    백그라운드에서 감정 분석을 수행하는 스레드
    VLM 추론이 느리므로 별도 스레드에서 처리
    """
    emotion_ready = pyqtSignal(int, str)  # face_index, emotion

    def __init__(self, face_detector):
        super().__init__()
        self.face_detector = face_detector
        self.running = True
        self.pending_analysis = None
        self.lock = threading.Lock()

    def request_analysis(self, frame, faces):
        """감정 분석 요청"""
        with self.lock:
            if faces:
                # 가장 큰 얼굴만 분석 (리소스 절약)
                largest_idx = 0
                largest_area = 0
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face['box']
                    area = (x2 - x1) * (y2 - y1)
                    if area > largest_area:
                        largest_area = area
                        largest_idx = i

                self.pending_analysis = (frame.copy(), faces[largest_idx]['box'], largest_idx)

    def run(self):
        while self.running:
            analysis_data = None

            with self.lock:
                if self.pending_analysis:
                    analysis_data = self.pending_analysis
                    self.pending_analysis = None

            if analysis_data:
                frame, face_box, face_idx = analysis_data
                emotion = self.face_detector.analyze_emotion_vlm(frame, face_box)
                if emotion:
                    self.emotion_ready.emit(face_idx, emotion)

            time.sleep(0.1)

    def stop(self):
        self.running = False


class SystemMonitor(QThread):
    stats_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        prev_idle, prev_total = 0, 0
        while self.running:
            stats = {}

            # CPU
            try:
                with open('/proc/stat', 'r') as f:
                    fields = f.readline().split()[1:]
                idle = int(fields[3])
                total = sum(int(x) for x in fields)

                diff_idle = idle - prev_idle
                diff_total = total - prev_total

                if diff_total > 0:
                    stats['cpu'] = int(100 * (1 - diff_idle / diff_total))
                else:
                    stats['cpu'] = 0

                prev_idle, prev_total = idle, total
            except:
                stats['cpu'] = 0

            # Temp
            try:
                max_temp = 0
                for i in range(6):
                    path = f'/sys/class/thermal/thermal_zone{i}/temp'
                    if os.path.exists(path):
                        with open(path) as f:
                            max_temp = max(max_temp, int(f.read()) // 1000)
                stats['temp'] = max_temp
            except:
                stats['temp'] = 0

            # RAM
            try:
                with open('/proc/meminfo') as f:
                    lines = f.readlines()
                total = int(lines[0].split()[1]) / 1024 / 1024
                avail = int(lines[2].split()[1]) / 1024 / 1024
                stats['ram_used'] = total - avail
                stats['ram_total'] = total
            except:
                stats['ram_used'] = 0
                stats['ram_total'] = 8

            # RK3588 GPU (Mali)
            try:
                with open('/sys/class/devfreq/fb000000.gpu/load', 'r') as f:
                    load_str = f.read().strip()  # "3@300000000Hz"
                    stats['gpu_load'] = int(load_str.split('@')[0])
                with open('/sys/class/devfreq/fb000000.gpu/cur_freq', 'r') as f:
                    gpu_freq = int(f.read().strip()) // 1000000  # MHz
                    stats['gpu_freq'] = gpu_freq
                    # Estimate GPU power: Mali G610 ~2-5W at full load
                    # Scale by frequency: 300MHz=0.5W, 1000MHz=5W
                    stats['gpu_power'] = round(0.5 + (gpu_freq - 300) * 4.5 / 700, 1)
            except:
                stats['gpu_load'] = 0
                stats['gpu_freq'] = 0
                stats['gpu_power'] = 0

            # RK3588 NPU - Check actual activity via trans_stat
            try:
                with open('/sys/class/devfreq/fdab0000.npu/cur_freq', 'r') as f:
                    rk_npu_freq = int(f.read().strip()) // 1000000  # MHz
                    stats['rk_npu_freq'] = rk_npu_freq
                # Estimate RK NPU power: ~1-3W range based on frequency
                # 300MHz=0.5W, 1000MHz=3W
                stats['rk_npu_power'] = round(0.5 + (rk_npu_freq - 300) * 2.5 / 700, 1)
            except:
                stats['rk_npu_freq'] = 0
                stats['rk_npu_power'] = 0

            # DX-M1 NPU (via dxrt-cli)
            try:
                import subprocess
                result = subprocess.run(['dxrt-cli', '-s'], capture_output=True, text=True, timeout=2)
                output = result.stdout
                # Parse "NPU 0: voltage 750 mV, clock 1000 MHz, temperature 41'C"
                temps = []
                voltages = []
                clocks = []
                for line in output.split('\n'):
                    if 'NPU' in line and 'temperature' in line:
                        # Temperature
                        temp_part = line.split('temperature')[1]
                        temp = int(temp_part.strip().split("'")[0])
                        temps.append(temp)
                        # Voltage
                        volt_part = line.split('voltage')[1].split('mV')[0]
                        volt = int(volt_part.strip())
                        voltages.append(volt)
                        # Clock
                        clock_part = line.split('clock')[1].split('MHz')[0]
                        clock = int(clock_part.strip())
                        clocks.append(clock)
                stats['dx_npu_temp'] = max(temps) if temps else 0
                stats['dx_npu_volt'] = voltages[0] if voltages else 0  # mV
                stats['dx_npu_clock'] = clocks[0] if clocks else 0  # MHz
                stats['dx_npu_count'] = len(temps)
                # Estimate power: P = V * I, assume ~0.5A per core at full load
                # 750mV * 0.5A * 3 cores ≈ 1.125W per NPU cluster
                if voltages and clocks:
                    # Rough estimate based on voltage and clock
                    estimated_power = (voltages[0] / 1000) * 0.5 * len(temps)  # Watts
                    stats['dx_npu_power'] = round(estimated_power, 1)
                else:
                    stats['dx_npu_power'] = 0
            except:
                stats['dx_npu_temp'] = 0
                stats['dx_npu_volt'] = 0
                stats['dx_npu_clock'] = 0
                stats['dx_npu_count'] = 0
                stats['dx_npu_power'] = 0

            self.stats_updated.emit(stats)
            time.sleep(1)

    def stop(self):
        self.running = False


# =============================================================================
# LLM 워커 스레드 (LLM Worker Thread)
# =============================================================================

class LLMWorker(QThread):
    """
    Vision LLM API 호출을 처리하는 백그라운드 워커 스레드

    이 클래스는 별도의 스레드에서 LLM API 호출을 수행하여
    메인 UI 스레드가 블로킹되지 않도록 합니다.

    주요 기능:
    - 비동기 LLM API 호출 (Gemini, GPT-4o, Claude 등)
    - 이미지 + 텍스트 프롬프트 처리
    - 실시간 통역 모드 지원
    - 객체 검출 결과 컨텍스트 전달

    Signals:
        response_ready (str, str, bool): LLM 응답 완료 시그널
            - text: 응답 텍스트
            - timestamp: 타임스탬프
            - is_user: 사용자 메시지 여부
        status_changed (str): 상태 변경 시그널

    Attributes:
        selected_model (str): 선택된 LLM 모델명
        translation_mode (bool): 통역 모드 활성화 여부
        translation_lang1/2 (str): 통역 언어 설정
    """

    # Qt 시그널 정의
    response_ready = pyqtSignal(str, str, bool)  # (응답텍스트, 시간, 사용자여부)
    status_changed = pyqtSignal(str)              # 상태 문자열

    def __init__(self):
        """LLMWorker 초기화"""
        super().__init__()
        self.running = True                         # 실행 상태 플래그
        self.queries = []                           # 처리할 쿼리 큐
        self.lock = threading.Lock()                # 쿼리 큐 접근 락
        self.detected_objects = []                  # 검출된 객체 목록
        self.det_boxes = []                         # 검출된 바운딩 박스
        self.obj_lock = threading.Lock()            # 객체 정보 접근 락
        self.current_frame = None                   # 현재 프레임 (LLM 분석용)
        self.frame_lock = threading.Lock()          # 프레임 접근 락
        self.selected_model = "Claude Sonnet"       # 기본 모델: Claude Sonnet

        # 통역 모드 설정
        self.translation_mode = False               # 통역 모드 비활성화
        self.translation_lang1 = "Korean"           # 언어 1: 한국어
        self.translation_lang2 = "English"

    def set_translation_mode(self, enabled, lang1="Korean", lang2="English"):
        """Set translation mode settings"""
        self.translation_mode = enabled
        self.translation_lang1 = lang1
        self.translation_lang2 = lang2

    def set_model(self, model_name):
        """Set the Vision LLM model to use"""
        self.selected_model = model_name

    def update_frame(self, frame):
        """Update the current frame for Vision LLM"""
        with self.frame_lock:
            self.current_frame = frame.copy() if frame is not None else None

    def update_detections(self, detections):
        """Update with full detection info including positions"""
        with self.obj_lock:
            self.detected_objects = [d['class'] for d in detections]
            self.det_boxes = [(d['class'], d['box'], d['score']) for d in detections]

    def add_query(self, msg):
        with self.lock:
            self.queries.append(msg)

    def run(self):
        while self.running:
            msg = None
            with self.lock:
                if self.queries:
                    msg = self.queries.pop(0)

            if msg:
                ts = datetime.now().strftime("%H:%M:%S")
                self.response_ready.emit(msg, ts, True)

                model_config = VISION_LLM_OPTIONS.get(self.selected_model, {})
                provider = model_config.get("provider", "local")

                self.status_changed.emit(f"Analyzing ({self.selected_model})...")

                # Get current frame for Vision LLM
                with self.frame_lock:
                    frame = self.current_frame.copy() if self.current_frame is not None else None

                # Build context with positions
                with self.obj_lock:
                    det_boxes = self.det_boxes.copy()

                if det_boxes:
                    obj_desc = []
                    for cls, box, score in det_boxes:
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        if cx < 213:
                            pos = "왼쪽"
                        elif cx < 427:
                            pos = "중앙"
                        else:
                            pos = "오른쪽"
                        obj_desc.append(f"{cls}({pos}, {score:.0%})")
                    context = f"YOLO 감지 결과: {', '.join(obj_desc)}. "
                else:
                    context = ""

                try:
                    if provider == "local":
                        # Use local RK3588 LLM (text only)
                        prompt = f"{context}\n\n질문: {msg}\n\n간결하게 한국어로 답변:"
                        resp = requests.post(
                            f"{RKLLAMA_API}/api/generate",
                            json={
                                "model": "qwen3-0.6b",
                                "prompt": prompt,
                                "stream": False,
                                "options": {"num_predict": 100, "temperature": 0.7}
                            },
                            timeout=30
                        )
                        if resp.status_code == 200:
                            result = resp.json().get('response', '').strip()
                            if '</think>' in result:
                                result = result.split('</think>')[-1].strip()
                        else:
                            result = f"Local LLM Error: {resp.status_code}"

                    elif provider in ["gemini", "groq", "claude", "openai", "xai", "local_vlm"] and frame is not None:
                        # Use Vision LLM API with image
                        image_base64 = VisionLLMClient.encode_image(frame)

                        # Check if translation mode is enabled
                        if self.translation_mode:
                            # Translation mode - translate between two languages
                            # Use English context for clarity
                            eng_context = context.replace("왼쪽", "left").replace("중앙", "center").replace("오른쪽", "right").replace("YOLO 감지 결과", "YOLO detection")

                            vision_prompt = f"""[TRANSLATION MODE]
You are a real-time interpreter between {self.translation_lang1} and {self.translation_lang2}.

RULES:
1. If input is in {self.translation_lang1} → Output MUST be in {self.translation_lang2} ONLY
2. If input is in {self.translation_lang2} → Output MUST be in {self.translation_lang1} ONLY
3. Output ONLY the translated text, nothing else
4. Do NOT include the original text
5. Do NOT add explanations or notes

Image context: {eng_context}

INPUT TO TRANSLATE: {msg}

TRANSLATION:"""
                            print(f"[Translation] Mode ON: {self.translation_lang1} ↔ {self.translation_lang2}")
                        else:
                            # Normal mode - respond in UI language
                            # Language code to English name mapping for LLM instruction
                            LANG_NAMES = {
                                "ko": "Korean",
                                "en": "English",
                                "ja": "Japanese",
                                "zh": "Chinese",
                                "es": "Spanish",
                                "fr": "French",
                                "de": "German",
                                "pt": "Portuguese",
                                "ru": "Russian",
                                "ar": "Arabic",
                            }
                            ui_lang_name = LANG_NAMES.get(current_ui_lang, "Korean")

                            vision_prompt = f"""{context}

User question: {msg}

CRITICAL INSTRUCTION: You MUST respond ONLY in {ui_lang_name}. Do NOT use any other language. Be concise (2-3 sentences max)."""

                        model = model_config.get("model", "")

                        if provider == "gemini":
                            result = VisionLLMClient.call_gemini(image_base64, vision_prompt, API_KEYS["gemini"])
                        elif provider == "groq":
                            result = VisionLLMClient.call_groq(image_base64, vision_prompt, API_KEYS["groq"])
                        elif provider == "claude":
                            result = VisionLLMClient.call_claude(image_base64, vision_prompt, API_KEYS["claude"], model)
                        elif provider == "openai":
                            result = VisionLLMClient.call_openai(image_base64, vision_prompt, API_KEYS["openai"], model)
                        elif provider == "xai":
                            result = VisionLLMClient.call_xai(image_base64, vision_prompt, API_KEYS["xai"], model)
                        # RK3588 로컬 VLM 모델 처리 (NPU 추론)
                        elif provider == "local_vlm":
                            result = VisionLLMClient.call_local_vlm(image_base64, vision_prompt, model)
                        else:
                            result = "Unknown provider"
                    else:
                        result = "No frame available or invalid provider"

                    ts = datetime.now().strftime("%H:%M:%S")
                    self.response_ready.emit(result, ts, False)
                    self.status_changed.emit("Ready")

                except Exception as e:
                    self.status_changed.emit("Error")
                    print(f"LLM Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                time.sleep(0.05)

    def stop(self):
        self.running = False


# =============================================================================
# 메인 애플리케이션 클래스 (Main Application Class)
# =============================================================================

class ProductionApp(QMainWindow):
    """
    듀얼 NPU 데모의 메인 윈도우 애플리케이션 클래스

    이 클래스는 전체 애플리케이션의 UI와 로직을 관리합니다.
    PyQt5 QMainWindow를 상속하여 GUI를 구성하고, 다양한 워커 스레드와
    상호작용하여 실시간 객체 검출 및 AI 채팅 기능을 제공합니다.

    주요 구성 요소:
    1. 비디오 패널 (왼쪽)
       - 실시간 카메라 영상 표시
       - YOLO 객체 검출 결과 오버레이
       - FPS, 추론 시간, 객체 수 표시

    2. 채팅 패널 (오른쪽 상단)
       - AI 비전 어시스턴트와 대화
       - 음성 입력(STT) / 출력(TTS) 지원
       - 퀵 버튼으로 빠른 질문

    3. 시스템 모니터 (오른쪽 하단)
       - CPU, GPU, NPU 사용량
       - 온도, 메모리 상태

    주요 기능:
    - 실시간 객체 검출 (DX-M1 NPU)
    - Vision LLM 채팅 (다중 프로바이더)
    - STT/TTS 음성 기능
    - 실시간 통역 모드
    - 자동 장면 설명
    - UI 다국어 지원

    Attributes:
        cap (cv2.VideoCapture): 카메라 캡처 객체
        detector (DXM1Detector): 객체 검출기
        llm_worker (LLMWorker): LLM 처리 워커
        tts_worker (TextToSpeech): 음성 출력 워커
        sys_monitor (SystemMonitor): 시스템 모니터 워커
    """

    def __init__(self):
        """
        ProductionApp 초기화

        UI 구성, 카메라 초기화, 검출기 초기화, 워커 스레드 시작을
        순차적으로 수행합니다.
        """
        super().__init__()

        # 윈도우 기본 설정
        self.setWindowTitle(f"Dual NPU Demo - DX-M1 ({YOLO_MODEL}) + RK3588 LLM")
        self.setGeometry(50, 50, 1400, 850)          # 위치(50,50), 크기(1400x850)
        self.setStyleSheet("background-color: #1a1a2e;")  # 다크 테마 배경

        # 카메라 및 검출 관련 변수
        self.cap = None                              # OpenCV 비디오 캡처 객체
        self.detector = None
        self.pose_detector = None                    # 포즈/제스처 감지기
        self.face_detector = None                    # 얼굴 검출기 (SCRFD)
        self.emotion_analyzer = None                 # 감정 분석 스레드
        self.last_emotion = None                     # 마지막 분석된 감정
        self.last_emotion_time = 0                   # 마지막 감정 분석 시간
        self.emotion_analysis_interval = 5.0         # 감정 분석 간격 (초)
        self.detection_mode = 'none'                 # 감지 모드: 'none', 'object', 'pose', 'face'
        self.current_gestures = []                   # 현재 감지된 제스처
        self.current_faces = []                      # 현재 감지된 얼굴
        self.detections = []
        self.frame_count = 0
        self.fps = 0
        self.fps_start = time.time()
        self.inference_time = 0
        self.current_result_frame = None  # For Vision LLM capture

        # Voice settings
        self.voice_output_enabled = True
        self.auto_desc_voice_output = True
        self.auto_desc_interval = 30
        self.auto_desc_enabled = False

        # Translation mode settings
        self.translation_mode = False
        self.translation_lang1 = "Korean"
        self.translation_lang2 = "English"
        self.translation_lang1_display = "한국어"
        self.translation_lang2_display = "English"

        # Track if chat has any messages (to avoid resetting on language change)
        self.chat_has_messages = False

        self.init_ui()
        self.init_camera()
        self.init_detector()
        self.init_workers()

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(1)  # As fast as possible

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title bar with settings button
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Production Dual NPU Demo")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setStyleSheet("color: #00d4ff; padding: 5px;")
        title_bar.addWidget(title)

        title_bar.addStretch()

        # Translation mode button
        self.translate_btn = QPushButton(get_text("translate_btn"))
        self.translate_btn.setStyleSheet("""
            QPushButton { background: #30363d; color: #c9d1d9; border: 1px solid #484f58; border-radius: 6px; padding: 6px 12px; font-size: 12px; }
            QPushButton:hover { background: #484f58; }
        """)
        self.translate_btn.clicked.connect(self.open_translation_settings)
        title_bar.addWidget(self.translate_btn)

        # Settings button
        self.settings_btn = QPushButton(get_text("settings_btn"))
        self.settings_btn.setStyleSheet("""
            QPushButton { background: #30363d; color: #c9d1d9; border: 1px solid #484f58; border-radius: 6px; padding: 6px 12px; font-size: 12px; }
            QPushButton:hover { background: #484f58; }
        """)
        self.settings_btn.clicked.connect(self.open_auto_desc_settings)
        title_bar.addWidget(self.settings_btn)

        layout.addLayout(title_bar)

        # Main content
        content = QHBoxLayout()

        # Left - Video
        left = self.create_video_panel()
        content.addWidget(left, 3)

        # Right - Chat + System
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        chat = self.create_chat_panel()
        right_layout.addWidget(chat, 3)

        sys_panel = self.create_system_panel()
        right_layout.addWidget(sys_panel, 1)

        content.addWidget(right, 2)
        layout.addLayout(content)

    def create_video_panel(self):
        self.video_panel = QGroupBox(get_text("dx_panel_title", model=YOLO_MODEL))
        panel = self.video_panel
        panel.setFont(QFont('Arial', 10, QFont.Bold))
        panel.setStyleSheet("""
            QGroupBox { color: #00ff88; border: 2px solid #00ff88; border-radius: 8px; margin-top: 8px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

        layout = QVBoxLayout(panel)

        # DX-M1 감지 모드 선택 바
        mode_bar = QFrame()
        mode_bar.setStyleSheet("background: #1a1a2e; border-radius: 4px; padding: 3px;")
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(8, 4, 8, 4)
        mode_layout.setSpacing(8)

        mode_title = QLabel("DX-M1 Mode:")
        mode_title.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 11px;")
        mode_layout.addWidget(mode_title)

        # 객체 감지 버튼
        self.object_btn = QPushButton("📦 Object")
        self.object_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: #7f8c8d;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #34495e; }
        """)
        self.object_btn.clicked.connect(self.toggle_object_detection)
        self.object_btn.setToolTip("YOLOX-S 객체 감지 (80 클래스)")
        mode_layout.addWidget(self.object_btn)

        # 포즈/제스처 버튼 (라이선스 문제로 비활성화)
        self.pose_btn = QPushButton("🦴 Pose/Gesture")
        self.pose_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a2e;
                color: #555555;
                border: 1px solid #333344;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #252538; color: #666666; }
        """)
        self.pose_btn.clicked.connect(self.toggle_pose_detection)
        self.pose_btn.setToolTip("⚠️ 비활성화: YOLOv5Pose AGPL-3.0 라이선스 문제")
        mode_layout.addWidget(self.pose_btn)

        # 얼굴/감정 버튼 - SCRFD 모델 (MIT 라이선스)
        self.face_btn = QPushButton("😊 Face/Emotion")
        self.face_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #e74c3c; color: white; }
        """)
        self.face_btn.clicked.connect(self.toggle_face_detection)
        self.face_btn.setToolTip("얼굴 검출 + VLM 감정 분석 (BSD/MIT 라이선스)")
        mode_layout.addWidget(self.face_btn)

        mode_layout.addStretch()

        # 현재 모드 표시
        self.mode_status = QLabel("Camera Only")
        self.mode_status.setStyleSheet("color: #888; font-size: 10px;")
        mode_layout.addWidget(self.mode_status)

        layout.addWidget(mode_bar)

        # 비디오 표시 영역
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #000; border: 1px solid #333; border-radius: 4px;")
        layout.addWidget(self.video_label)

        # Info bar
        info = QFrame()
        info.setStyleSheet("background: #16213e; border-radius: 4px; padding: 5px;")
        info_layout = QHBoxLayout(info)
        info_layout.setContentsMargins(5, 2, 5, 2)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        info_layout.addWidget(self.fps_label)

        self.inf_time_label = QLabel("Inf: -- ms")
        self.inf_time_label.setStyleSheet("color: #ffaa00;")
        info_layout.addWidget(self.inf_time_label)

        self.det_count_label = QLabel("Objects: 0")
        self.det_count_label.setStyleSheet("color: #ff6b6b;")
        info_layout.addWidget(self.det_count_label)

        info_layout.addStretch()

        self.npu_label = QLabel("NPU: --")
        self.npu_label.setStyleSheet("color: #888;")
        info_layout.addWidget(self.npu_label)

        layout.addWidget(info)

        # Detection list
        self.det_list_label = QLabel("Detected: None")
        self.det_list_label.setStyleSheet("color: #aaa; padding: 5px;")
        self.det_list_label.setWordWrap(True)
        layout.addWidget(self.det_list_label)

        return panel

    def create_chat_panel(self):
        self.chat_panel = QGroupBox(get_text("llm_panel_title"))
        panel = self.chat_panel
        panel.setFont(QFont('Arial', 10, QFont.Bold))
        panel.setStyleSheet("""
            QGroupBox { color: #58a6ff; border: 2px solid #58a6ff; border-radius: 8px; margin-top: 8px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

        layout = QVBoxLayout(panel)

        # Model Selector and Status Row
        model_row = QHBoxLayout()

        # Status
        self.llm_status = QLabel("LLM: Initializing...")
        self.llm_status.setStyleSheet("color: #888; padding: 3px;")
        model_row.addWidget(self.llm_status)

        model_row.addStretch()

        # Model Selector ComboBox
        self.model_label = QLabel(get_text("model") + ":")
        self.model_label.setStyleSheet("color: #8b949e; font-size: 10px;")
        model_row.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(160)
        self.model_combo.setStyleSheet("""
            QComboBox { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; }
            QComboBox:hover { border-color: #58a6ff; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #21262d; color: #c9d1d9; selection-background-color: #58a6ff; }
        """)
        for model_name in VISION_LLM_OPTIONS.keys():
            cost = VISION_LLM_OPTIONS[model_name].get("cost", "")
            self.model_combo.addItem(f"{model_name}")
        # Set Claude Sonnet as default model
        self.model_combo.setCurrentText("Claude Sonnet")
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_row.addWidget(self.model_combo)

        layout.addLayout(model_row)

        # Chat display - Modern messenger style
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont('Malgun Gothic', 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: #0d1117;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        self.update_chat_placeholder()
        layout.addWidget(self.chat_display)

        # Input
        input_layout = QHBoxLayout()

        # Voice input button
        self.voice_btn = QPushButton("🎤")
        self.voice_btn.setFixedSize(40, 40)
        self.voice_btn.setToolTip("Hold to record voice (STT)")
        self.voice_btn.setStyleSheet("""
            QPushButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 20px; font-size: 16px; }
            QPushButton:hover { background: #30363d; border-color: #58a6ff; }
            QPushButton:pressed { background: #ff6b6b; }
        """)
        self.voice_btn.pressed.connect(self.start_voice_input)
        self.voice_btn.released.connect(self.stop_voice_input)
        input_layout.addWidget(self.voice_btn)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText(get_text("input_placeholder"))
        self.chat_input.setFont(QFont('Malgun Gothic', 11))
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 20px;
                padding: 10px 16px;
            }
            QLineEdit:focus {
                border-color: #58a6ff;
                background: #0d1117;
            }
        """)
        self.chat_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.chat_input)

        send_btn = QPushButton("➤")
        send_btn.setFixedSize(40, 40)
        send_btn.setStyleSheet("""
            QPushButton {
                background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #2ea043; }
            QPushButton:pressed { background: #1a7f37; }
        """)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)

        # Voice output toggle
        self.tts_btn = QPushButton("🔊")
        self.tts_btn.setFixedSize(40, 40)
        self.tts_btn.setCheckable(True)
        self.tts_btn.setChecked(True)
        self.tts_btn.setToolTip("Toggle voice output (TTS)")
        self.tts_btn.setStyleSheet("""
            QPushButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 20px; font-size: 16px; }
            QPushButton:hover { background: #30363d; }
            QPushButton:checked { background: #238636; border-color: #238636; }
        """)
        self.tts_btn.toggled.connect(self.toggle_voice_output)
        input_layout.addWidget(self.tts_btn)

        layout.addLayout(input_layout)

        # Quick buttons with stored references
        quick = QHBoxLayout()
        self.quick_btns = []
        # key for button text, prompt_key for the actual prompt to send
        quick_btn_data = [
            ("what_see", "what_see_prompt"),
            ("analyze", "analyze_prompt"),
            ("explain", "explain_prompt")
        ]
        for key, prompt_key in quick_btn_data:
            btn = QPushButton(get_text(key))
            btn.setProperty("trans_key", key)
            btn.setProperty("prompt_key", prompt_key)
            btn.setStyleSheet("QPushButton { background: #21262d; color: #8b949e; border: 1px solid #30363d; border-radius: 4px; padding: 6px; } QPushButton:hover { background: #30363d; }")
            btn.clicked.connect(lambda _, pk=prompt_key: self.quick_ask(get_text(pk)))
            quick.addWidget(btn)
            self.quick_btns.append(btn)

        # Auto description settings button
        self.auto_desc_btn = QPushButton(get_text("auto"))
        self.auto_desc_btn.setToolTip("자동 설명 (클릭: 설정/끄기)")
        self.auto_desc_btn.setStyleSheet("""
            QPushButton { background: #21262d; color: #888; border: 1px solid #30363d; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background: #30363d; }
        """)
        self.auto_desc_btn.setCheckable(True)
        self.auto_desc_btn.clicked.connect(self.toggle_auto_description)
        quick.addWidget(self.auto_desc_btn)

        layout.addLayout(quick)

        return panel

    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if hasattr(self, 'llm_worker'):
            self.llm_worker.set_model(model_name)
            config = VISION_LLM_OPTIONS.get(model_name, {})
            provider = config.get("provider", "local")
            cost = config.get("cost", "")
            model = config.get("model", "")

            # Update status with API key check
            # 로컬 모델 (local, local_vlm)은 API 키 불필요
            if provider not in ("local", "local_vlm"):
                api_key = API_KEYS.get(provider, "")
                if api_key:
                    self.llm_status.setText(f"LLM: {model_name} ({cost})")
                    self.llm_status.setStyleSheet("color: #00ff88;")
                else:
                    self.llm_status.setText(f"LLM: {model_name} - {get_text('api_key_missing')}")
                    self.llm_status.setStyleSheet("color: #ff6b6b;")
            else:
                # 로컬 모델은 API 키 없이 바로 사용 가능
                self.llm_status.setText(f"LLM: {model_name} ({cost})")
                self.llm_status.setStyleSheet("color: #00ff88;")

            print(f"[LLM] Model changed to: {model_name} ({provider}/{model})")

    def create_system_panel(self):
        self.system_panel = QGroupBox(get_text("system_monitor"))
        panel = self.system_panel
        panel.setFont(QFont('Arial', 9, QFont.Bold))
        panel.setStyleSheet("""
            QGroupBox { color: #a371f7; border: 1px solid #a371f7; border-radius: 6px; margin-top: 6px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

        main_layout = QVBoxLayout(panel)
        main_layout.setSpacing(4)

        # Store stat labels for translation
        self.stat_labels = {}

        # Row 1: CPU, TEMP, RAM
        row1 = QHBoxLayout()
        for trans_key, color in [("cpu", "#00ff88"), ("temp", "#ffaa00"), ("ram", "#58a6ff")]:
            frame, lbl = self._create_stat_widget(trans_key, color)
            self.stat_labels[trans_key] = lbl
            row1.addWidget(frame)
        main_layout.addLayout(row1)

        # Row 2: GPU, RK-NPU, DX-NPU
        row2 = QHBoxLayout()
        for trans_key, color in [("gpu", "#ff6b9d"), ("rk_npu", "#00d4ff"), ("dx_npu", "#ffd700")]:
            frame, lbl = self._create_stat_widget(trans_key, color)
            self.stat_labels[trans_key] = lbl
            row2.addWidget(frame)
        main_layout.addLayout(row2)

        return panel

    def _create_stat_widget(self, trans_key, color):
        """Create a stat display widget with bar and value"""
        frame = QFrame()
        frame.setStyleSheet("background: #16213e; border-radius: 4px; padding: 2px;")
        v = QVBoxLayout(frame)
        v.setSpacing(1)
        v.setContentsMargins(4, 2, 4, 2)

        display_name = get_text(trans_key)
        lbl = QLabel(display_name)
        lbl.setProperty("trans_key", trans_key)
        lbl.setStyleSheet(f"color: {color}; font-size: 8px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(lbl)

        bar = QProgressBar()
        bar.setFixedHeight(10)
        bar.setTextVisible(False)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        v.addWidget(bar)

        val = QLabel("--")
        val.setStyleSheet("color: #888; font-size: 8px;")
        val.setAlignment(Qt.AlignCenter)
        v.addWidget(val)

        # Store references using trans_key
        attr_name = trans_key.lower().replace("-", "_")
        setattr(self, f"{attr_name}_bar", bar)
        setattr(self, f"{attr_name}_val", val)

        return frame, lbl

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("[Camera] Opened successfully")
        else:
            print("[Camera] Failed to open")

    def init_detector(self):
        # 객체 감지기 초기화 (YOLOX-S)
        self.detector = DXM1Detector(MODEL_PATH)
        if self.detector.initialize():
            self.npu_label.setText("NPU: Active")
            self.npu_label.setStyleSheet("color: #00ff88;")
        else:
            self.npu_label.setText("NPU: Error")
            self.npu_label.setStyleSheet("color: #ff0000;")

        # 포즈/제스처 감지기 초기화 (YOLOv5Pose) - 지연 로딩
        # 모드 전환 시 초기화됨
        self.pose_detector = None

    def init_pose_detector(self):
        """포즈 감지기 초기화 (모드 전환 시 호출)"""
        if self.pose_detector is None:
            pose_model_path = "/home/orangepi/model_for_demo/YOLOv5Pose640_1.dxnn"
            self.pose_detector = DXM1PoseDetector(pose_model_path)
            if self.pose_detector.initialize():
                print("[Pose] Detector initialized successfully")
                return True
            else:
                print("[Pose] Detector initialization failed")
                self.pose_detector = None
                return False
        return True

    def set_detection_mode(self, mode):
        """
        감지 모드 설정

        Args:
            mode: 'none', 'object', 'pose', 'face'
        """
        # 버튼 스타일 초기화
        btn_style_off = """
            QPushButton {
                background-color: #2c3e50;
                color: #7f8c8d;
                border: 1px solid #34495e;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #34495e; }
        """
        btn_style_on = """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """

        # 모든 버튼 비활성화 스타일
        if hasattr(self, 'object_btn'):
            self.object_btn.setStyleSheet(btn_style_off)
        if hasattr(self, 'pose_btn'):
            self.pose_btn.setStyleSheet(btn_style_off)
        if hasattr(self, 'face_btn'):
            self.face_btn.setStyleSheet(btn_style_off)

        if mode == 'none':
            self.detection_mode = 'none'
            if hasattr(self, 'mode_status'):
                self.mode_status.setText("Camera Only")
                self.mode_status.setStyleSheet("color: #888; font-size: 10px;")
            print("[Mode] Camera only (no detection)")

        elif mode == 'object':
            if self.detector and self.detector.initialized:
                self.detection_mode = 'object'
                if hasattr(self, 'object_btn'):
                    self.object_btn.setStyleSheet(btn_style_on)
                if hasattr(self, 'mode_status'):
                    self.mode_status.setText("Object Detection ON")
                    self.mode_status.setStyleSheet("color: #27ae60; font-size: 10px; font-weight: bold;")
                print("[Mode] Object detection ON")
            else:
                print("[Mode] Object detector not initialized")

        elif mode == 'pose':
            if self.init_pose_detector():
                self.detection_mode = 'pose'
                if hasattr(self, 'pose_btn'):
                    self.pose_btn.setStyleSheet(btn_style_on)
                if hasattr(self, 'mode_status'):
                    self.mode_status.setText("Pose/Gesture ON")
                    self.mode_status.setStyleSheet("color: #9b59b6; font-size: 10px; font-weight: bold;")
                print("[Mode] Pose/Gesture detection ON")
            else:
                print("[Mode] Failed to initialize pose detector")

        elif mode == 'face':
            # SCRFD 얼굴 검출기 사용 (MIT 라이선스)
            self.detection_mode = 'face'
            if hasattr(self, 'face_btn'):
                self.face_btn.setStyleSheet(btn_style_on)
            if hasattr(self, 'mode_status'):
                self.mode_status.setText("Face Detection ON")
                self.mode_status.setStyleSheet("color: #e74c3c; font-size: 10px; font-weight: bold;")
            print("[Mode] Face detection ON - SCRFD (MIT License)")

    def toggle_object_detection(self):
        """객체 감지 토글"""
        if self.detection_mode == 'object':
            self.set_detection_mode('none')
        else:
            self.set_detection_mode('object')

    def toggle_pose_detection(self):
        """포즈/제스처 감지 토글 - 현재 비활성화 (라이선스 문제)

        YOLOv5Pose 모델은 AGPL-3.0 라이선스로 상업적 사용에 제약이 있습니다.
        Apache 2.0 라이선스의 대안 모델(YOLOX-Pose, RTMPose)이 DX-M1용으로 준비되면 활성화됩니다.
        """
        # 라이선스 경고 메시지 표시
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("기능 비활성화")
        msg.setText("포즈/제스처 인식 기능이 비활성화되어 있습니다.")
        msg.setInformativeText(
            "YOLOv5Pose 모델은 AGPL-3.0 라이선스로 상업적 사용에 제약이 있습니다.\n\n"
            "Apache 2.0 라이선스의 대안 모델(YOLOX-Pose, RTMPose)이 "
            "DX-M1 NPU용으로 준비되면 이 기능이 활성화됩니다."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def toggle_face_detection(self):
        """얼굴/감정 감지 토글 - OpenCV Haar Cascade + VLM 감정 분석"""
        if self.detection_mode == 'face':
            self.set_detection_mode('none')
            # 감정 분석 스레드 중지
            if self.emotion_analyzer:
                self.emotion_analyzer.stop()
                self.emotion_analyzer.wait()
                self.emotion_analyzer = None
        else:
            # 얼굴 검출기 초기화 (필요한 경우)
            if self.face_detector is None:
                self.face_detector = DXM1FaceDetector()
                if not self.face_detector.initialize():
                    from PyQt5.QtWidgets import QMessageBox
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("초기화 실패")
                    msg.setText("얼굴 검출기 초기화에 실패했습니다.")
                    msg.setInformativeText("OpenCV Haar Cascade 로드에 실패했습니다.")
                    msg.exec_()
                    self.face_detector = None
                    return

            # 감정 분석 스레드 초기화
            if self.emotion_analyzer is None:
                self.emotion_analyzer = EmotionAnalyzerThread(self.face_detector)
                self.emotion_analyzer.emotion_ready.connect(self.on_emotion_ready)
                self.emotion_analyzer.start()
                print("[Emotion] Analyzer thread started")

            self.set_detection_mode('face')

    def on_emotion_ready(self, face_idx, emotion):
        """감정 분석 결과 콜백"""
        self.last_emotion = emotion
        self.last_emotion_time = time.time()
        print(f"[Emotion] Face {face_idx+1}: {emotion}")
        # 현재 얼굴 목록에 감정 업데이트
        if face_idx < len(self.current_faces):
            self.current_faces[face_idx]['emotion'] = emotion

    def init_workers(self):
        self.sys_monitor = SystemMonitor()
        self.sys_monitor.stats_updated.connect(self.on_stats)
        self.sys_monitor.start()

        self.llm_worker = LLMWorker()
        self.llm_worker.response_ready.connect(self.on_response)
        self.llm_worker.status_changed.connect(self.on_llm_status)
        self.llm_worker.start()

        # Initialize STT (Speech-to-Text)
        openai_key = API_KEYS.get('openai', '')
        if AUDIO_AVAILABLE and openai_key:
            self.stt_worker = SpeechToText(openai_key)
            self.stt_worker.transcription_ready.connect(self.on_transcription)
            self.stt_worker.status_changed.connect(self.on_stt_status)
            self.stt_worker.recording_state.connect(self.on_recording_state)
            print("[STT] Initialized with OpenAI Whisper")
        else:
            self.stt_worker = None
            print("[STT] Not available (missing audio libs or API key)")

        # Initialize TTS (Text-to-Speech)
        if AUDIO_AVAILABLE and openai_key:
            self.tts_worker = TextToSpeech(openai_key)
            self.tts_worker.status_changed.connect(self.on_tts_status)
            self.tts_worker.start()
            print("[TTS] Initialized with OpenAI TTS")
        else:
            self.tts_worker = None
            print("[TTS] Not available (missing audio libs or API key)")

        # Initialize Auto Description Worker
        self.auto_desc_worker = AutoDescriptionWorker(self.auto_desc_interval)
        self.auto_desc_worker.description_request.connect(self.on_auto_description_request)
        self.auto_desc_worker.start()

        # Initialize with the currently selected model in combo box
        current_model = self.model_combo.currentText()
        self.on_model_changed(current_model)

    def process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip horizontally to match coordinate system
        frame = cv2.flip(frame, 1)

        self.frame_count += 1
        result_frame = frame

        # 감지 모드에 따른 처리
        if self.detection_mode == 'none':
            # 카메라만 표시 (감지 없음)
            self.inference_time = 0
            self.detections = []
            self.current_gestures = []
            self.det_count_label.setText("Mode: Camera Only")
            self.det_list_label.setText("Detection: OFF")

        elif self.detection_mode == 'object':
            # 객체 감지 모드
            if self.detector and self.detector.initialized:
                t0 = time.time()
                detections, result_frame = self.detector.detect(frame)
                self.inference_time = (time.time() - t0) * 1000

                self.detections = detections
                self.llm_worker.update_detections(detections)

                self.det_count_label.setText(f"Objects: {len(detections)}")
                if detections:
                    counts = Counter([d['class'] for d in detections])
                    det_str = ", ".join([f"{k}:{v}" for k, v in counts.most_common(5)])
                    self.det_list_label.setText(f"Detected: {det_str}")
                else:
                    self.det_list_label.setText("Detected: None")

        elif self.detection_mode == 'pose':
            # 포즈/제스처 감지 모드
            if self.pose_detector and self.pose_detector.initialized:
                t0 = time.time()
                poses, result_frame, gestures = self.pose_detector.detect(frame)
                self.inference_time = (time.time() - t0) * 1000

                self.current_gestures = gestures
                self.det_count_label.setText(f"Poses: {len(poses)}")

                if gestures:
                    gesture_strs = [g['gesture'].replace('_', ' ').title() for g in gestures]
                    self.det_list_label.setText(f"Gestures: {', '.join(gesture_strs)}")
                else:
                    self.det_list_label.setText("Gestures: None")

        elif self.detection_mode == 'face':
            # 얼굴/감정 감지 모드 - OpenCV Haar Cascade + VLM 감정 분석
            if self.face_detector and self.face_detector.initialized:
                t0 = time.time()
                faces, result_frame = self.face_detector.detect(frame)
                self.inference_time = (time.time() - t0) * 1000

                # 이전 감정 결과 유지 (캐싱)
                if self.last_emotion and len(faces) > 0:
                    # 첫 번째 얼굴에 마지막 감정 적용 (5초 이내인 경우)
                    if time.time() - self.last_emotion_time < self.emotion_analysis_interval + 2:
                        faces[0]['emotion'] = self.last_emotion

                self.current_faces = faces
                self.det_count_label.setText(f"Faces: {len(faces)}")

                if faces:
                    # 주기적으로 감정 분석 요청
                    if (self.emotion_analyzer and
                        time.time() - self.last_emotion_time >= self.emotion_analysis_interval):
                        self.emotion_analyzer.request_analysis(frame, faces)

                    face_info = []
                    for i, face in enumerate(faces):
                        score = face['score']
                        emotion = face.get('emotion', '')
                        info = f"Face{i+1}"
                        if emotion:
                            info += f"-{emotion}"
                        face_info.append(info)
                    self.det_list_label.setText(", ".join(face_info[:3]))

                    # 결과 프레임 재생성 (감정 정보 포함)
                    result_frame = self.face_detector.visualize(frame, faces)
                else:
                    self.det_list_label.setText("Faces: None")
            else:
                self.det_count_label.setText("Mode: Face")
                self.det_list_label.setText("Initializing...")

        # Store current frame for Vision LLM capture
        self.current_result_frame = result_frame.copy()

        # Display
        rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_fps(self):
        elapsed = time.time() - self.fps_start
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        self.frame_count = 0
        self.fps_start = time.time()

        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        self.inf_time_label.setText(f"Inf: {self.inference_time:.1f}ms")

    def send_message(self):
        msg = self.chat_input.text().strip()
        if msg:
            self.chat_input.clear()
            # Capture current frame at the moment of sending
            self.capture_and_send(msg)

    def quick_ask(self, prompt):
        # Capture current frame at the moment of asking
        self.capture_and_send(prompt)

    def capture_and_send(self, msg):
        """Capture current YOLO frame and send to LLM"""
        # Get the current frame with YOLO detections
        if hasattr(self, 'current_result_frame') and self.current_result_frame is not None:
            self.llm_worker.update_frame(self.current_result_frame)
            self.llm_status.setText("Frame captured! Analyzing...")
            self.llm_status.setStyleSheet("color: #ffaa00;")
        else:
            self.llm_status.setText("No frame available")
            self.llm_status.setStyleSheet("color: #ff6b6b;")

        self.llm_worker.add_query(msg)

    def on_response(self, text, ts, is_user):
        if is_user:
            # User message - right aligned, green bubble
            html = f"""
            <div style='margin: 8px 0; text-align: right;'>
                <div style='display: inline-block; max-width: 85%; text-align: left;'>
                    <div style='background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
                                color: white; padding: 10px 14px; border-radius: 18px 18px 4px 18px;
                                font-size: 13px; line-height: 1.4;'>{text}</div>
                    <div style='color: #484f58; font-size: 10px; margin-top: 3px; text-align: right;'>{ts}</div>
                </div>
            </div>
            """
        else:
            # AI response - left aligned, dark bubble with icon
            html = f"""
            <div style='margin: 8px 0; text-align: left;'>
                <div style='display: inline-block; max-width: 85%;'>
                    <div style='color: #58a6ff; font-size: 10px; margin-bottom: 3px;'>🤖 AI</div>
                    <div style='background: #161b22; color: #e6edf3; padding: 10px 14px;
                                border-radius: 18px 18px 18px 4px; border: 1px solid #30363d;
                                font-size: 13px; line-height: 1.5;'>{text}</div>
                    <div style='color: #484f58; font-size: 10px; margin-top: 3px;'>{ts}</div>
                </div>
            </div>
            """
            # Speak the response if voice output is enabled
            if self.voice_output_enabled or getattr(self, '_auto_desc_pending', False):
                self.speak_response(text)
                self._auto_desc_pending = False
        self.chat_display.append(html)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        self.chat_has_messages = True  # Mark that chat has messages

    def on_llm_status(self, status):
        # Preserve model info in status display
        current_model = self.model_combo.currentText() if hasattr(self, 'model_combo') else ""
        if status == "Ready":
            config = VISION_LLM_OPTIONS.get(current_model, {})
            cost = config.get("cost", "")
            self.llm_status.setText(f"LLM: {current_model} ({cost})")
            self.llm_status.setStyleSheet("color: #00ff88;")
        elif status == "Error":
            self.llm_status.setText(f"LLM: {current_model} - Error")
            self.llm_status.setStyleSheet("color: #ff6b6b;")
        else:
            self.llm_status.setText(f"LLM: {status}")
            self.llm_status.setStyleSheet("color: #ffaa00;")

    def on_stats(self, stats):
        # CPU
        self.cpu_bar.setValue(stats.get('cpu', 0))
        self.cpu_val.setText(f"{stats.get('cpu', 0)}%")

        # Temperature
        self.temp_bar.setValue(min(100, stats.get('temp', 0)))
        self.temp_val.setText(f"{stats.get('temp', 0)}°C")

        # RAM
        used = stats.get('ram_used', 0)
        total = stats.get('ram_total', 1)
        self.ram_bar.setValue(int(100 * used / total))
        self.ram_val.setText(f"{used:.1f}G")

        # GPU (Mali) - Show load, freq, power
        gpu_load = stats.get('gpu_load', 0)
        gpu_freq = stats.get('gpu_freq', 0)
        gpu_power = stats.get('gpu_power', 0)
        self.gpu_bar.setValue(gpu_load)
        self.gpu_val.setText(f"{gpu_load}%/{gpu_freq}M/~{gpu_power}W")

        # RK3588 NPU - Show frequency and power estimate
        rk_freq = stats.get('rk_npu_freq', 0)
        rk_power = stats.get('rk_npu_power', 0)
        # Show frequency as bar (300-1000MHz range)
        freq_pct = int((rk_freq - 300) / 7) if rk_freq > 300 else 0  # 300-1000 -> 0-100%
        self.rk_npu_bar.setValue(min(100, freq_pct))
        self.rk_npu_val.setText(f"{rk_freq}M/~{rk_power}W")

        # DX-M1 NPU - Show temp, voltage, estimated power
        dx_temp = stats.get('dx_npu_temp', 0)
        dx_volt = stats.get('dx_npu_volt', 0)
        dx_power = stats.get('dx_npu_power', 0)
        dx_count = stats.get('dx_npu_count', 0)
        # Temperature as percentage (0-100°C range)
        self.dx_npu_bar.setValue(min(100, dx_temp))
        self.dx_npu_val.setText(f"{dx_temp}°C/{dx_volt}mV/~{dx_power}W")

    # ===== Voice Input (STT) Methods =====

    def start_voice_input(self):
        """Start voice recording"""
        if self.stt_worker:
            self.voice_btn.setStyleSheet("""
                QPushButton { background: #ff6b6b; color: white; border: 2px solid #ff6b6b; border-radius: 20px; font-size: 16px; }
            """)
            self.stt_worker.start_recording()
        else:
            self.llm_status.setText("STT not available")
            self.llm_status.setStyleSheet("color: #ff6b6b;")

    def stop_voice_input(self):
        """Stop voice recording"""
        if self.stt_worker:
            self.voice_btn.setStyleSheet("""
                QPushButton { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 20px; font-size: 16px; }
                QPushButton:hover { background: #30363d; border-color: #58a6ff; }
            """)
            self.stt_worker.stop_recording()

    def on_transcription(self, text):
        """Handle transcribed text from STT"""
        if text:
            self.chat_input.setText(text)
            # Auto-send the transcribed message
            self.send_message()

    def on_stt_status(self, status):
        """Handle STT status updates"""
        self.llm_status.setText(f"STT: {status}")
        if "error" in status.lower():
            self.llm_status.setStyleSheet("color: #ff6b6b;")
        elif "recording" in status.lower():
            self.llm_status.setStyleSheet("color: #ff6b6b;")
        else:
            self.llm_status.setStyleSheet("color: #00ff88;")

    def on_recording_state(self, is_recording):
        """Update UI based on recording state"""
        if is_recording:
            self.voice_btn.setText("🔴")
        else:
            self.voice_btn.setText("🎤")

    # ===== Voice Output (TTS) Methods =====

    def toggle_voice_output(self, enabled):
        """Toggle TTS voice output"""
        self.voice_output_enabled = enabled
        if enabled:
            self.tts_btn.setText("🔊")
        else:
            self.tts_btn.setText("🔇")

    def on_tts_status(self, status):
        """Handle TTS status updates"""
        if "speaking" in status.lower():
            self.llm_status.setText("🔊 Speaking...")
            self.llm_status.setStyleSheet("color: #58a6ff;")
        elif "error" in status.lower():
            self.llm_status.setStyleSheet("color: #ff6b6b;")

    def speak_response(self, text):
        """Speak the response using TTS"""
        if self.voice_output_enabled and self.tts_worker:
            # Clean text for TTS (remove special characters, emojis)
            clean_text = text.replace('\n', ' ').strip()
            if clean_text:
                self.tts_worker.speak(clean_text)

    # ===== Translation Mode Methods =====

    def open_translation_settings(self):
        """Toggle translation mode or open settings dialog"""
        # If translation mode is ON, turn it OFF immediately
        if self.translation_mode:
            self.translation_mode = False
            if hasattr(self, 'llm_worker') and self.llm_worker:
                self.llm_worker.set_translation_mode(False, "", "")
            self.translate_btn.setStyleSheet("""
                QPushButton { background: #30363d; color: #c9d1d9; border: 1px solid #484f58; border-radius: 6px; padding: 6px 12px; font-size: 12px; }
                QPushButton:hover { background: #484f58; }
            """)
            self.translate_btn.setText("🌐 통역")
            self.llm_status.setText("통역 모드 OFF")
            self.llm_status.setStyleSheet("color: #888;")
            print("[Translation] Mode turned OFF")
            return

        # If OFF, open settings dialog
        print("[Translation] Opening settings dialog...")
        current_settings = {
            'enabled': self.translation_mode,
            'lang1': self.translation_lang1,
            'lang2': self.translation_lang2,
        }
        dialog = TranslationSettingsDialog(self, current_settings)

        result = dialog.exec_()
        print(f"[Translation] Dialog result: {result} (Accepted={QDialog.Accepted})")

        if result == QDialog.Accepted:
            settings = dialog.get_settings()
            print(f"[Translation] Settings: {settings}")
            self.translation_mode = settings['enabled']
            self.translation_lang1 = settings['lang1']
            self.translation_lang2 = settings['lang2']
            self.translation_lang1_display = settings['lang1_display']
            self.translation_lang2_display = settings['lang2_display']

            print(f"[Translation] Mode={self.translation_mode}, {self.translation_lang1} ↔ {self.translation_lang2}")

            # Update LLM worker translation settings
            if hasattr(self, 'llm_worker') and self.llm_worker:
                self.llm_worker.set_translation_mode(
                    self.translation_mode,
                    self.translation_lang1,
                    self.translation_lang2
                )

            # Update button style
            if self.translation_mode:
                self.translate_btn.setStyleSheet("""
                    QPushButton { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 6px; padding: 6px 12px; font-size: 12px; }
                    QPushButton:hover { background: #2ea043; }
                """)
                self.translate_btn.setText(f"🌐 {self.translation_lang1_display}↔{self.translation_lang2_display}")
                self.llm_status.setText(f"🌐 통역: {self.translation_lang1_display} ↔ {self.translation_lang2_display}")
                self.llm_status.setStyleSheet("color: #58a6ff;")
                print(f"[Translation] UI Updated - Button: {self.translate_btn.text()}")
            else:
                self.translate_btn.setStyleSheet("""
                    QPushButton { background: #30363d; color: #c9d1d9; border: 1px solid #484f58; border-radius: 6px; padding: 6px 12px; font-size: 12px; }
                    QPushButton:hover { background: #484f58; }
                """)
                self.translate_btn.setText("🌐 통역")
                print("[Translation] UI Updated - Mode OFF")

    # ===== Auto Description Methods =====

    def toggle_auto_description(self):
        """Toggle auto description on/off"""
        self.auto_desc_enabled = not self.auto_desc_enabled
        self.auto_desc_worker.set_enabled(self.auto_desc_enabled)
        self.auto_desc_btn.setChecked(self.auto_desc_enabled)

        if self.auto_desc_enabled:
            # Turning ON - update style and trigger first description immediately
            self.auto_desc_btn.setStyleSheet("""
                QPushButton { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 4px; padding: 6px; }
                QPushButton:hover { background: #2ea043; }
            """)
            self.llm_status.setText(f"Auto desc: ON ({self.auto_desc_interval}s)")
            self.llm_status.setStyleSheet("color: #ffaa00;")
            print(f"[AutoDesc] Turned ON (interval={self.auto_desc_interval}s)")
            # Trigger first description immediately
            QTimer.singleShot(500, self.on_auto_description_request)
        else:
            # Turning OFF
            self.auto_desc_btn.setStyleSheet("""
                QPushButton { background: #21262d; color: #888; border: 1px solid #30363d; border-radius: 4px; padding: 6px; }
                QPushButton:hover { background: #30363d; }
            """)
            self.llm_status.setText("Auto desc: OFF")
            self.llm_status.setStyleSheet("color: #888;")
            print("[AutoDesc] Turned OFF")

    def open_auto_desc_settings(self):
        """Open auto description settings dialog"""
        global current_ui_lang
        dialog = AutoDescriptionSettingsDialog(
            self,
            current_interval=self.auto_desc_interval,
            current_enabled=self.auto_desc_enabled,
            current_ui_language=current_ui_lang
        )

        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            was_enabled = self.auto_desc_enabled
            self.auto_desc_enabled = settings['enabled']
            self.auto_desc_interval = settings['interval']
            self.auto_desc_voice_output = settings['voice_output']

            # Handle UI language change
            new_lang = settings.get('ui_language', 'ko')
            if new_lang != current_ui_lang:
                current_ui_lang = new_lang
                self.apply_translations()
                print(f"[UI] Language changed to: {new_lang}")

            print(f"[AutoDesc] Settings applied: enabled={self.auto_desc_enabled}, interval={self.auto_desc_interval}s, voice={self.auto_desc_voice_output}")

            # Update workers
            self.auto_desc_worker.set_interval(self.auto_desc_interval)
            self.auto_desc_worker.set_enabled(self.auto_desc_enabled)

            # Update button state and style
            self.auto_desc_btn.setChecked(self.auto_desc_enabled)

            if self.auto_desc_enabled:
                # Active state - green/highlight color
                self.auto_desc_btn.setStyleSheet("""
                    QPushButton { background: #238636; color: white; border: 1px solid #2ea043; border-radius: 4px; padding: 6px; }
                    QPushButton:hover { background: #2ea043; }
                """)
                self.llm_status.setText(f"Auto desc: ON ({self.auto_desc_interval}s)")
                self.llm_status.setStyleSheet("color: #ffaa00;")
                # If just enabled from settings, trigger first description immediately
                if not was_enabled:
                    QTimer.singleShot(500, self.on_auto_description_request)
            else:
                # Inactive state
                self.auto_desc_btn.setStyleSheet("""
                    QPushButton { background: #21262d; color: #888; border: 1px solid #30363d; border-radius: 4px; padding: 6px; }
                    QPushButton:hover { background: #30363d; }
                """)
                self.llm_status.setText("Auto desc: OFF")
                self.llm_status.setStyleSheet("color: #888;")

    def on_auto_description_request(self):
        """Handle auto description request"""
        print(f"[AutoDesc] Request triggered, interval={self.auto_desc_interval}s")

        # Check if LLM worker is available
        if not hasattr(self, 'llm_worker') or self.llm_worker is None:
            print("[AutoDesc] LLM worker not available")
            return

        # Capture and analyze current scene
        if hasattr(self, 'current_result_frame') and self.current_result_frame is not None:
            self.llm_worker.update_frame(self.current_result_frame)
            print("[AutoDesc] Frame updated for LLM")

            # Use a specific prompt for auto-description
            auto_prompt = f"⏱️ {get_text('auto_desc_prompt')}"

            # Set flag for voice output on response
            self._auto_desc_pending = True

            # Send query - LLMWorker will handle displaying the message
            self.llm_worker.add_query(auto_prompt)
        else:
            print("[AutoDesc] No frame available for auto description")

    def update_chat_placeholder(self):
        """Update chat display with translated placeholder"""
        self.chat_display.setHtml(f"""
            <div style='text-align:center; padding: 20px;'>
                <p style='color:#586069; font-size: 12px;'>{get_text('ai_assistant')}</p>
                <p style='color:#484f58; font-size: 11px;'>{get_text('ai_hint')}</p>
            </div>
        """)

    def apply_translations(self):
        """Apply translations to all UI elements when language changes"""
        global current_ui_lang
        print(f"[UI] Applying translations for language: {current_ui_lang}")

        # Update buttons
        self.translate_btn.setText(get_text("translate_btn"))
        self.settings_btn.setText(get_text("settings_btn"))
        self.auto_desc_btn.setText(get_text("auto"))

        # Update quick buttons
        for btn in self.quick_btns:
            key = btn.property("trans_key")
            if key:
                btn.setText(get_text(key))

        # Update panel titles
        self.video_panel.setTitle(get_text("dx_panel_title", model=YOLO_MODEL))
        self.chat_panel.setTitle(get_text("llm_panel_title"))
        self.system_panel.setTitle(get_text("system_monitor"))

        # Update labels
        self.model_label.setText(get_text("model") + ":")
        self.chat_input.setPlaceholderText(get_text("input_placeholder"))

        # Update stat labels
        for trans_key, lbl in self.stat_labels.items():
            lbl.setText(get_text(trans_key))

        # Update chat placeholder only if chat has no messages
        if not self.chat_has_messages:
            self.update_chat_placeholder()

        print(f"[UI] Translations applied successfully")

    def closeEvent(self, event):
        self.sys_monitor.stop()
        self.sys_monitor.quit()
        self.sys_monitor.wait()

        self.llm_worker.stop()
        self.llm_worker.quit()
        self.llm_worker.wait()

        # Stop TTS worker
        if self.tts_worker:
            self.tts_worker.stop()
            self.tts_worker.quit()
            self.tts_worker.wait()

        # Stop Auto Description worker
        if hasattr(self, 'auto_desc_worker'):
            self.auto_desc_worker.stop()
            self.auto_desc_worker.quit()
            self.auto_desc_worker.wait()

        # Stop Emotion Analyzer thread
        if self.emotion_analyzer:
            self.emotion_analyzer.stop()
            self.emotion_analyzer.quit()
            self.emotion_analyzer.wait()

        if self.cap:
            self.cap.release()

        event.accept()


# =============================================================================
# 메인 함수 (Main Entry Point)
# =============================================================================

def main():
    """
    애플리케이션 메인 진입점

    PyQt5 애플리케이션을 초기화하고 메인 윈도우를 생성하여 실행합니다.
    다크 테마 팔레트를 적용하고, Fusion 스타일을 사용합니다.

    실행 방법:
        $ python3 production_app.py

    또는:
        $ DISPLAY=:0.0 python3 production_app.py  (SSH 원격 실행 시)
    """
    # Qt 애플리케이션 인스턴스 생성
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 크로스 플랫폼 스타일

    # 다크 테마 팔레트 설정
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 46))       # 배경색
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200)) # 텍스트색
    app.setPalette(palette)

    # 메인 윈도우 생성 및 표시
    window = ProductionApp()
    window.show()

    # 이벤트 루프 실행 (종료까지 블로킹)
    sys.exit(app.exec_())


# =============================================================================
# 스크립트 직접 실행 시 진입점
# =============================================================================
if __name__ == "__main__":
    main()
