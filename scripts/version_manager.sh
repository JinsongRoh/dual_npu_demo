#!/bin/bash
# ================================================================================
# 버전 관리 도구 (Version Manager)
# ================================================================================
# 회사명: MetaVu Co., Ltd.
# 개발자: JINSONG ROH
# 설명: 백업, 롤백, 비교, Git 관리 통합 도구
# ================================================================================

PROJECT_DIR="/home/orangepi/dual_npu_demo"
BACKUP_DIR="$PROJECT_DIR/backups"
MAIN_FILE="$PROJECT_DIR/production_app.py"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

show_menu() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    버전 관리 도구 v1.0                        ║"
    echo "║                    MetaVu Co., Ltd.                          ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║                                                              ║"
    echo "║   [1] 백업 생성          - 현재 버전 타임스탬프 백업          ║"
    echo "║   [2] 백업 목록          - 저장된 백업 파일 확인             ║"
    echo "║   [3] 버전 롤백          - 이전 버전으로 복원                ║"
    echo "║   [4] 버전 비교          - 현재와 백업 버전 비교             ║"
    echo "║   [5] Git 커밋           - 현재 변경사항 Git 커밋            ║"
    echo "║   [6] Git 로그           - Git 커밋 이력 확인                ║"
    echo "║   [7] Git 롤백           - Git 특정 커밋으로 복원            ║"
    echo "║   [8] 전체 상태          - 프로젝트 상태 요약                ║"
    echo "║   [9] 오래된 백업 정리   - 30일 이상 된 백업 삭제            ║"
    echo "║   [0] 종료                                                   ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -n " 메뉴를 선택하세요: "
}

# 1. 백업 생성
do_backup() {
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_FILE="$BACKUP_DIR/production_app_v${TIMESTAMP}.py"
    mkdir -p "$BACKUP_DIR"

    if cp "$MAIN_FILE" "$BACKUP_FILE"; then
        echo ""
        echo -e "${GREEN} 백업 완료: $(basename "$BACKUP_FILE")${NC}"
        echo " 파일 크기: $(du -h "$BACKUP_FILE" | cut -f1)"
    else
        echo -e "${RED} 백업 실패!${NC}"
    fi
}

# 2. 백업 목록
list_backups() {
    echo ""
    echo -e "${YELLOW} 백업 파일 목록:${NC}"
    echo ""
    ls -lht "$BACKUP_DIR"/*.py 2>/dev/null | head -20 | while read line; do
        echo "   $line"
    done
    TOTAL=$(ls "$BACKUP_DIR"/*.py 2>/dev/null | wc -l)
    echo ""
    echo -e " 총 ${GREEN}${TOTAL}${NC}개의 백업 파일"
}

# 3. 버전 롤백
do_rollback() {
    bash "$PROJECT_DIR/scripts/rollback.sh"
}

# 4. 버전 비교
compare_versions() {
    echo ""
    echo -e "${YELLOW} 비교할 백업 파일 선택:${NC}"
    echo ""

    BACKUPS=($(ls -t "$BACKUP_DIR"/*.py 2>/dev/null))

    if [ ${#BACKUPS[@]} -eq 0 ]; then
        echo -e "${RED} 백업 파일이 없습니다.${NC}"
        return
    fi

    for i in "${!BACKUPS[@]}"; do
        echo "  [$((i+1))] $(basename "${BACKUPS[$i]}")"
    done

    echo ""
    echo -n " 번호 선택: "
    read CHOICE

    if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le ${#BACKUPS[@]} ]; then
        SELECTED="${BACKUPS[$((CHOICE-1))]}"
        echo ""
        echo -e "${CYAN} 변경사항 (현재 vs $(basename "$SELECTED")):${NC}"
        echo "─────────────────────────────────────────────────────────"
        diff --color=always -u "$SELECTED" "$MAIN_FILE" | head -50
        echo ""
        echo " (처음 50줄만 표시, 전체 확인: diff -u $SELECTED $MAIN_FILE)"
    else
        echo -e "${RED} 잘못된 선택${NC}"
    fi
}

# 5. Git 커밋
do_git_commit() {
    cd "$PROJECT_DIR"
    echo ""
    echo -e "${YELLOW} 현재 변경사항:${NC}"
    git status --short
    echo ""
    echo -n " 커밋 메시지를 입력하세요: "
    read COMMIT_MSG

    if [ -n "$COMMIT_MSG" ]; then
        git add .
        git commit -m "$COMMIT_MSG"
        echo ""
        echo -e "${GREEN} 커밋 완료!${NC}"
    else
        echo -e "${RED} 커밋이 취소되었습니다.${NC}"
    fi
}

# 6. Git 로그
show_git_log() {
    cd "$PROJECT_DIR"
    echo ""
    echo -e "${YELLOW} Git 커밋 이력 (최근 15개):${NC}"
    echo ""
    git log --oneline --decorate -15 2>/dev/null || echo -e "${RED} Git 이력이 없습니다.${NC}"
}

# 7. Git 롤백
do_git_rollback() {
    cd "$PROJECT_DIR"
    echo ""
    echo -e "${YELLOW} Git 커밋 이력:${NC}"
    git log --oneline -10 2>/dev/null
    echo ""
    echo -n " 복원할 커밋 해시를 입력하세요 (앞 7자리): "
    read COMMIT_HASH

    if [ -n "$COMMIT_HASH" ]; then
        echo ""
        echo -n " production_app.py만 복원하시겠습니까? (y/n): "
        read CONFIRM

        if [ "$CONFIRM" == "y" ] || [ "$CONFIRM" == "Y" ]; then
            # 현재 버전 먼저 백업
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            cp "$MAIN_FILE" "$BACKUP_DIR/production_app_before_git_rollback_${TIMESTAMP}.py"

            git checkout "$COMMIT_HASH" -- production_app.py
            echo ""
            echo -e "${GREEN} 복원 완료! (롤백 전 버전이 백업되었습니다)${NC}"
        fi
    fi
}

# 8. 전체 상태
show_status() {
    cd "$PROJECT_DIR"
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW} 프로젝트 상태 요약${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e " ${GREEN}메인 파일:${NC}"
    echo "   production_app.py: $(du -h "$MAIN_FILE" | cut -f1)"
    echo "   최종 수정: $(stat -c %y "$MAIN_FILE" | cut -d'.' -f1)"
    echo ""
    echo -e " ${GREEN}백업 현황:${NC}"
    BACKUP_COUNT=$(ls "$BACKUP_DIR"/*.py 2>/dev/null | wc -l)
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    echo "   백업 파일 수: ${BACKUP_COUNT}개"
    echo "   백업 폴더 크기: ${BACKUP_SIZE:-0}"
    echo ""
    echo -e " ${GREEN}Git 상태:${NC}"
    git status --short 2>/dev/null || echo "   Git 변경사항 없음"
    echo ""
    COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    echo "   총 커밋 수: ${COMMIT_COUNT}"
    echo ""
}

# 9. 오래된 백업 정리
cleanup_old_backups() {
    echo ""
    echo -e "${YELLOW} 30일 이상 된 백업 파일:${NC}"
    OLD_FILES=$(find "$BACKUP_DIR" -name "*.py" -mtime +30 2>/dev/null)

    if [ -z "$OLD_FILES" ]; then
        echo "   삭제할 파일이 없습니다."
        return
    fi

    echo "$OLD_FILES" | while read file; do
        echo "   $(basename "$file")"
    done

    echo ""
    echo -n " 위 파일들을 삭제하시겠습니까? (y/n): "
    read CONFIRM

    if [ "$CONFIRM" == "y" ] || [ "$CONFIRM" == "Y" ]; then
        find "$BACKUP_DIR" -name "*.py" -mtime +30 -delete
        echo -e "${GREEN} 정리 완료!${NC}"
    fi
}

# 메인 루프
while true; do
    show_menu
    read CHOICE

    case $CHOICE in
        1) do_backup ;;
        2) list_backups ;;
        3) do_rollback ;;
        4) compare_versions ;;
        5) do_git_commit ;;
        6) show_git_log ;;
        7) do_git_rollback ;;
        8) show_status ;;
        9) cleanup_old_backups ;;
        0) echo ""; echo -e "${GREEN} 종료합니다.${NC}"; exit 0 ;;
        *) echo -e "${RED} 잘못된 선택입니다.${NC}" ;;
    esac

    echo ""
    echo -n " 계속하려면 Enter를 누르세요..."
    read
done
