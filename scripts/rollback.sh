#!/bin/bash
# ================================================================================
# 롤백 스크립트 (Rollback Script)
# ================================================================================
# 회사명: MetaVu Co., Ltd.
# 개발자: JINSONG ROH
# 설명: 백업 파일에서 production_app.py 복원
# ================================================================================

PROJECT_DIR="/home/orangepi/dual_npu_demo"
BACKUP_DIR="$PROJECT_DIR/backups"
MAIN_FILE="$PROJECT_DIR/production_app.py"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}========================================"
echo " 버전 롤백 도구 (Version Rollback Tool)"
echo "========================================${NC}"
echo ""

# 백업 파일 목록 확인
BACKUPS=($(ls -t "$BACKUP_DIR"/*.py 2>/dev/null))

if [ ${#BACKUPS[@]} -eq 0 ]; then
    echo -e "${RED} 오류: 백업 파일이 없습니다.${NC}"
    echo " 먼저 ./scripts/backup.sh 를 실행하여 백업을 생성하세요."
    exit 1
fi

echo -e "${YELLOW} 사용 가능한 백업 파일:${NC}"
echo ""

for i in "${!BACKUPS[@]}"; do
    BACKUP_FILE="${BACKUPS[$i]}"
    FILENAME=$(basename "$BACKUP_FILE")
    FILESIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    FILEDATE=$(stat -c %y "$BACKUP_FILE" | cut -d'.' -f1)
    echo -e "  ${GREEN}[$((i+1))]${NC} $FILENAME"
    echo "      크기: $FILESIZE | 날짜: $FILEDATE"
    echo ""
done

echo "  [0] 취소"
echo ""
echo -n " 복원할 버전 번호를 선택하세요: "
read CHOICE

# 입력 검증
if [ "$CHOICE" == "0" ]; then
    echo ""
    echo -e "${YELLOW} 롤백이 취소되었습니다.${NC}"
    exit 0
fi

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt ${#BACKUPS[@]} ]; then
    echo ""
    echo -e "${RED} 오류: 잘못된 선택입니다.${NC}"
    exit 1
fi

SELECTED_BACKUP="${BACKUPS[$((CHOICE-1))]}"
SELECTED_NAME=$(basename "$SELECTED_BACKUP")

echo ""
echo -e "${YELLOW} 선택된 백업: $SELECTED_NAME${NC}"
echo ""
echo -n " 정말로 복원하시겠습니까? 현재 파일은 백업됩니다. (y/n): "
read CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo ""
    echo -e "${YELLOW} 롤백이 취소되었습니다.${NC}"
    exit 0
fi

# 현재 파일 백업 (롤백 전)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BEFORE_ROLLBACK="$BACKUP_DIR/production_app_before_rollback_${TIMESTAMP}.py"
cp "$MAIN_FILE" "$BEFORE_ROLLBACK"
echo ""
echo -e "${GREEN} 현재 버전 백업 완료: $(basename "$BEFORE_ROLLBACK")${NC}"

# 롤백 실행
cp "$SELECTED_BACKUP" "$MAIN_FILE"

echo ""
echo -e "${GREEN}========================================"
echo " 롤백 완료!"
echo "========================================"
echo " 복원된 버전: $SELECTED_NAME"
echo " 복원 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================${NC}"
echo ""
echo " 롤백을 취소하려면 다음 명령을 실행하세요:"
echo "   cp $BEFORE_ROLLBACK $MAIN_FILE"
echo ""
