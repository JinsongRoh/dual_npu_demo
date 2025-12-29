dual_npu_demo프로젝트를 개발 중이야. "/home/orangepi/dual_npu_demo/" 프로젝트 폴더의 README.md, DEVLOG.md, DEV_GUIDELINES.md, .dev.vars 파일 등을 참조하여 개발을 계속 이어 진행해.


● TTS 오디오 출력 정상 작동 확인!

  최종 수정 사항 요약:
  | 항목       | 설정                   |
  |------------|------------------------|
  | 출력 장치  | HDMI1 직접 (device 2)  |
  | 입력 장치  | LifeCam USB (device 4) |
  | 샘플레이트 | 48000Hz 리샘플링       |
  | 채널       | 모노→스테레오 변환     |

  이제 Production Demo 앱에서 TTS 소리가 정상 출력됩니다!

● 오디오 문제 모두 해결되었습니다!

  최종 설정:
  | 항목 | 설정                        |
  |------|-----------------------------|
  | 출력 | HDMI1 (hw:2,0) - HDMI2 포트 |
  | 입력 | LifeCam USB 마이크          |
  | TTS  | 48000Hz + 스테레오 변환     |

  앱 코드(production_app.py)에 저장되어 재부팅 후에도 유지됩니다.

> 재부팅 후에도 오디오 설정 유지되는지 확인해줘 

● 현재 설정 파일들을 확인하겠습니다.

● Bash(cat ~/.config/pulse/default.pa)
  ⎿  .include /etc/pulse/default.pa                          

     # Load default native protocol socket (fix for GUI apps)
     … +7 lines (ctrl+o to expand)

● Bash(grep -A2 "sd.default" /home/orangepi/dual_npu_demo/production_app.py |
      head -5)
  ⎿      sd.default.device = (4, 2)   
         sd.default.samplerate = 48000
         AUDIO_AVAILABLE = True
     except ImportError:

● 재부팅 후 유지되는 설정:

  | 파일                       | 설정                                    | 상태      |
  |----------------------------|-----------------------------------------|-----------|
  | ~/.config/pulse/default.pa | PulseAudio 기본 출력: HDMI1             | ✅ 저장됨 |
  | ~/.config/pulse/default.pa | PulseAudio 기본 입력: USB 마이크        | ✅ 저장됨 |
  | production_app.py          | sounddevice 출력: HDMI1 (device 2)      | ✅ 저장됨 |
  | production_app.py          | sounddevice 입력: USB 마이크 (device 4) | ✅ 저장됨 |
  | production_app.py          | 샘플레이트: 48000Hz                     | ✅ 저장됨 |

  모든 설정이 파일에 저장되어 재부팅 후에도 유지됩니다.

