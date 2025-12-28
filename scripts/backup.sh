#!/bin/bash
# ================================================================================
# 백업 스크립트 (Backup Script)
# ================================================================================
# 회사명: MetaVu Co., Ltd.
# 개발자: JINSONG ROH
# 설명: production_app.py의 타임스탬프 백업 생성
# ================================================================================

PROJECT_DIR="/home/orangepi/dual_npu_demo"
BACKUP_DIR="$PROJECT_DIR/backups"
MAIN_FILE="$PROJECT_DIR/production_app.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/production_app_v${TIMESTAMP}.py"

# 백업 폴더 생성 (없으면)
mkdir -p "$BACKUP_DIR"

# 백업 생성
if [ -f "$MAIN_FILE" ]; then
    cp "$MAIN_FILE" "$BACKUP_FILE"
    echo "========================================"
    echo " 백업 완료!"
    echo "========================================"
    echo " 백업 파일: $BACKUP_FILE"
    echo " 파일 크기: $(du -h "$BACKUP_FILE" | cut -f1)"
    echo " 생성 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # 최근 백업 목록 표시
    echo ""
    echo " 최근 백업 목록 (최신 5개):"
    ls -lt "$BACKUP_DIR"/*.py 2>/dev/null | head -5 | while read line; do
        echo "   $line"
    done
    echo "========================================"
else
    echo " 오류: $MAIN_FILE 파일을 찾을 수 없습니다."
    exit 1
fi
