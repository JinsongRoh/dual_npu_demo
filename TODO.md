# TODO List - Dual NPU Demo

> 마지막 업데이트: 2025-12-31

## 완료된 항목 (Completed)

- [x] RKNPU 0.9.8 드라이버 테스트
- [x] DX-M1 NPU 테스트 (정상 동작)
- [x] RKLLM 1.2.1 런타임 설치
- [x] 텍스트 LLM 테스트 (Qwen3-0.6B, 31 tokens/s)
- [x] 애플리케이션 실행 테스트
- [x] Git 원격 저장소 동기화
- [x] TTS (Text-to-Speech) 수정 완료

## 진행 중 (In Progress)

- [x] STT (Speech-to-Text) 테스트 ✅ 완료 (2025-12-30)
  - OpenAI Whisper API 연동 완료
  - 마이크 녹음 테스트 성공
  - 앱 내 STT 정상 동작 확인

## 알려진 이슈 (Known Issues)

- [x] NPU 메모리 충돌 ✅ 해결됨 (2025-12-30)
  - DX-M1 + RKLLM VLM 동시 사용 정상 동작 확인

## 최근 완료 (2025-12-29 00:30)

- [x] STT 구현 확인
  - SpeechToText 클래스 (OpenAI Whisper 연동)
  - 마이크 버튼 (🎤) Push-to-Talk 방식
  - 녹음 → 전사 → 자동 전송 워크플로우
  - 다국어 지원 (한/영/일/중)

- [x] TTS (Text-to-Speech) 버그 수정
  - OpenAI API stream_to_file 디프리케이션 버그 수정
  - with_streaming_response.create() 패턴으로 변경
  - 오디오 출력 정상 동작 확인

- [x] 다중 VLM 모델 지원 추가
  - qwen2.5-vl-3b: 3B 파라미터, Vision 4초, LLM 8 tokens/s
  - qwen2-vl-2b: 2B 파라미터, Vision 3초, LLM 12 tokens/s
  - RK3588_VLM_MODELS 설정 딕셔너리 추가
  - 모델별 이미지 크기 및 토큰 설정 지원

- [x] production_app.py에 RK3588 VLM 직접 추론 연동
  - RK3588LocalVLM 싱글톤 클래스 추가
  - rkllm_binding.py, ztu_somemodelruntime_rknnlite2.py 복사
  - call_local_vlm() 메서드에서 직접 NPU 추론 사용
  - 테스트 결과: Vision Encoder 4.2초, LLM 응답 4.3초

- [x] VLM 모델 1.2.1 버전용으로 업데이트
  - Qwen2.5-VL-3B-Instruct-RKLLM 다운로드 완료 (4.95GB)
  - Vision Encoder: 4.5초
  - LLM: ~8 tokens/s
  - librknnrt 2.3.2 설치 완료

## 대기 중 (Pending)

- [x] 제스처 인식 기능 구현 ✅ 완료 (2025-12-30)
  - DXM1PoseDetector 클래스 추가 (YOLOv5Pose 모델)
  - GestureRecognizer 클래스 추가 (8가지 제스처)
  - UI 감지 모드 전환 버튼 추가
  - ⚠️ **라이선스 문제로 비활성화됨** (YOLOv5Pose AGPL-3.0)

- [ ] Apache 2.0 포즈 모델 변환 ⬅️ 다음 작업
  - YOLOX-Pose 또는 RTMPose DX-M1 변환 필요
  - 변환 후 포즈/제스처 기능 재활성화

- [x] 얼굴 인식 구현 ✅ 완료 (2025-12-31)
  - DXM1FaceDetector 클래스 추가 (OpenCV Haar Cascade)
  - BSD 라이선스 (상업적 사용 가능)
  - 다중 얼굴 동시 검출 지원
  - UI Face/Emotion 버튼 연동

- [x] 감정 분석 기능 추가 ✅ 완료 (2025-12-31)
  - VLM (Qwen2.5-VL-3B) 연동 감정 분석
  - 7가지 감정 분류: happy, sad, angry, surprised, neutral, fear, disgust
  - EmotionAnalyzerThread 백그라운드 처리
  - 5초 간격 주기적 분석

- [ ] 다중 카메라 지원

## 참고 사항

### VLM 모델 정보
- 모델: happyme531/Qwen2.5-VL-3B-Instruct-RKLLM
- RKLLM 버전: 1.2.1
- 드라이버 버전: 0.9.8
- 예상 성능: Vision Encoder 3.4s + LLM 8.2 tokens/s

### 모델 파일 위치
- RKLLM 모델: /mnt/external/rkllm_models/
- DX-M1 모델: /home/orangepi/model_for_demo/
