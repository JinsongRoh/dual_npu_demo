#!/bin/bash
# 화면 녹화 스크립트
# 사용법: ./record_screen.sh [출력파일명]

OUTPUT="${1:-demo_$(date +%Y%m%d_%H%M%S).mp4}"
DISPLAY_ENV=":0.0"

echo "========================================"
echo "  화면 녹화 시작"
echo "========================================"
echo "출력 파일: $OUTPUT"
echo "중지: Ctrl+C 또는 q 키"
echo "========================================"

# 화면 해상도 확인
RESOLUTION=$(DISPLAY=$DISPLAY_ENV xdpyinfo | grep dimensions | awk '{print $2}')
echo "해상도: $RESOLUTION"
echo ""

# 녹화 시작 (오디오 포함)
DISPLAY=$DISPLAY_ENV ffmpeg -y \
    -f x11grab -framerate 30 -video_size $RESOLUTION -i $DISPLAY_ENV \
    -f pulse -i default \
    -c:v libx264 -preset ultrafast -crf 23 \
    -c:a aac -b:a 128k \
    -pix_fmt yuv420p \
    "$OUTPUT"

echo ""
echo "녹화 완료: $OUTPUT"
ls -lh "$OUTPUT"
