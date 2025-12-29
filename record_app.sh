#!/bin/bash
# 앱 창만 녹화하는 스크립트 (이름으로 검색)
# 사용법: ./record_app.sh [출력파일명]

OUTPUT="${1:-app_demo_$(date +%Y%m%d_%H%M%S).mp4}"
DISPLAY_ENV=":0.0"

echo "========================================"
echo "  앱 창 녹화"
echo "========================================"

# Production 창 찾기
WINDOW_IDS=$(DISPLAY=$DISPLAY_ENV xdotool search --name "Production" 2>/dev/null)
if [ -z "$WINDOW_IDS" ]; then
    echo "❌ 앱 창을 찾을 수 없습니다!"
    echo "앱이 실행 중인지 확인하세요."
    exit 1
fi

# 가장 큰 창 찾기
MAX_AREA=0
WINDOW_ID=""
for WID in $WINDOW_IDS; do
    SIZE=$(DISPLAY=$DISPLAY_ENV xdotool getwindowgeometry $WID 2>/dev/null | grep "Geometry" | awk '{print $2}')
    W=$(echo $SIZE | cut -d'x' -f1)
    H=$(echo $SIZE | cut -d'x' -f2)
    if [ -n "$W" ] && [ -n "$H" ]; then
        AREA=$((W * H))
        if [ $AREA -gt $MAX_AREA ]; then
            MAX_AREA=$AREA
            WINDOW_ID=$WID
            BEST_SIZE=$SIZE
        fi
    fi
done

# 창이 너무 작으면 전체 화면 녹화
if [ $MAX_AREA -lt 10000 ]; then
    echo "⚠️  메인 창을 찾지 못해 전체 화면 녹화합니다."
    WIDTH=1920
    HEIGHT=1080
    X_POS=0
    Y_POS=0
else
    GEOMETRY=$(DISPLAY=$DISPLAY_ENV xdotool getwindowgeometry $WINDOW_ID)
    POS=$(echo "$GEOMETRY" | grep "Position" | awk '{print $2}' | cut -d'(' -f1)
    SIZE=$(echo "$GEOMETRY" | grep "Geometry" | awk '{print $2}')
    X_POS=$(echo $POS | cut -d',' -f1)
    Y_POS=$(echo $POS | cut -d',' -f2)
    WIDTH=$(echo $SIZE | cut -d'x' -f1)
    HEIGHT=$(echo $SIZE | cut -d'x' -f2)
fi

echo "위치: ${X_POS},${Y_POS}"
echo "크기: ${WIDTH}x${HEIGHT}"
echo "출력: $OUTPUT"
echo "중지: Ctrl+C 또는 q"
echo "========================================"

# 녹화 시작
DISPLAY=$DISPLAY_ENV ffmpeg -y \
    -f x11grab -framerate 30 -video_size ${WIDTH}x${HEIGHT} -i ${DISPLAY_ENV}+${X_POS},${Y_POS} \
    -f pulse -i default \
    -c:v libx264 -preset ultrafast -crf 23 \
    -c:a aac -b:a 128k \
    -pix_fmt yuv420p \
    "$OUTPUT"

echo ""
echo "✅ 녹화 완료: $OUTPUT"
ls -lh "$OUTPUT"
