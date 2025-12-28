#!/bin/bash
# ================================================================================
# 보안 검사 스크립트 (Security Check Script)
# ================================================================================
# 회사명: MetaVu Co., Ltd.
# 설명: 프로젝트 전체에서 민감 정보 및 보안 취약점을 검사
# ================================================================================

PROJECT_DIR="/home/orangepi/dual_npu_demo"
cd "$PROJECT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    보안 검사 도구 v1.0                        ║"
echo "║                    MetaVu Co., Ltd.                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

ISSUES_FOUND=0
WARNINGS=0

# ==========================================
# 1. API 키 하드코딩 검사
# ==========================================
echo -e "${YELLOW}[1/6] API 키 하드코딩 검사...${NC}"

API_PATTERNS=(
    "sk-[a-zA-Z0-9]\{20,\}"
    "sk-proj-[a-zA-Z0-9]\{20,\}"
    "sk-ant-[a-zA-Z0-9]\{20,\}"
    "AIza[a-zA-Z0-9_-]\{35\}"
    "AKIA[A-Z0-9]\{16\}"
)

for pattern in "${API_PATTERNS[@]}"; do
    MATCHES=$(grep -rn --include="*.py" "$pattern" . 2>/dev/null | grep -v ".git" | grep -v "__pycache__")
    if [ -n "$MATCHES" ]; then
        echo -e "${RED}  [위험] API 키 패턴 발견:${NC}"
        echo "$MATCHES" | while read line; do
            echo "    $line"
        done
        ((ISSUES_FOUND++))
    fi
done

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}  [통과] API 키 하드코딩 없음${NC}"
fi

# ==========================================
# 2. 비밀번호 하드코딩 검사
# ==========================================
echo -e "${YELLOW}[2/6] 비밀번호 하드코딩 검사...${NC}"

PASSWORD_MATCHES=$(grep -rn --include="*.py" -iE "(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]" . 2>/dev/null | grep -v ".git" | grep -v "__pycache__" | grep -v "environ" | grep -v "getenv" | grep -v "# " | grep -v ".md")

if [ -n "$PASSWORD_MATCHES" ]; then
    echo -e "${RED}  [위험] 비밀번호 하드코딩 의심:${NC}"
    echo "$PASSWORD_MATCHES" | head -10 | while read line; do
        echo "    $line"
    done
    ((ISSUES_FOUND++))
else
    echo -e "${GREEN}  [통과] 비밀번호 하드코딩 없음${NC}"
fi

# ==========================================
# 3. .gitignore 검사
# ==========================================
echo -e "${YELLOW}[3/6] .gitignore 설정 검사...${NC}"

REQUIRED_IGNORES=(".env" ".dev.vars" "*.key" "*.pem" "backups/")
MISSING_IGNORES=()

for item in "${REQUIRED_IGNORES[@]}"; do
    if ! grep -q "$item" .gitignore 2>/dev/null; then
        MISSING_IGNORES+=("$item")
    fi
done

if [ ${#MISSING_IGNORES[@]} -gt 0 ]; then
    echo -e "${YELLOW}  [경고] .gitignore에 추가 권장:${NC}"
    for item in "${MISSING_IGNORES[@]}"; do
        echo "    - $item"
    done
    ((WARNINGS++))
else
    echo -e "${GREEN}  [통과] .gitignore 설정 양호${NC}"
fi

# ==========================================
# 4. 민감 파일 존재 검사
# ==========================================
echo -e "${YELLOW}[4/6] 민감 파일 검사...${NC}"

SENSITIVE_FILES=("credentials.json" "service-account.json" "id_rsa" ".pem" ".key")
FOUND_SENSITIVE=0

for pattern in "${SENSITIVE_FILES[@]}"; do
    FOUND=$(find . -name "*$pattern*" -type f 2>/dev/null | grep -v ".git")
    if [ -n "$FOUND" ]; then
        echo -e "${YELLOW}  [경고] 민감 파일 발견:${NC}"
        echo "$FOUND" | while read f; do
            echo "    $f"
        done
        ((WARNINGS++))
        FOUND_SENSITIVE=1
    fi
done

if [ $FOUND_SENSITIVE -eq 0 ]; then
    echo -e "${GREEN}  [통과] 민감 파일 없음${NC}"
fi

# ==========================================
# 5. Git 스테이징 영역 검사
# ==========================================
echo -e "${YELLOW}[5/6] Git 스테이징 영역 검사...${NC}"

STAGED=$(git diff --cached --name-only 2>/dev/null)
STAGED_SENSITIVE=0

if [ -n "$STAGED" ]; then
    for file in $STAGED; do
        if [[ "$file" == *".env"* ]] || [[ "$file" == *".vars"* ]] || [[ "$file" == *"credential"* ]]; then
            echo -e "${RED}  [위험] 민감 파일 스테이징됨: $file${NC}"
            ((ISSUES_FOUND++))
            STAGED_SENSITIVE=1
        fi
    done
fi

if [ $STAGED_SENSITIVE -eq 0 ]; then
    echo -e "${GREEN}  [통과] 스테이징 영역 안전${NC}"
fi

# ==========================================
# 6. 환경 변수 사용 검사
# ==========================================
echo -e "${YELLOW}[6/6] 환경 변수 사용 패턴 검사...${NC}"

ENV_USAGE=$(grep -rn --include="*.py" "os.environ\|getenv\|load_dotenv" . 2>/dev/null | grep -v ".git" | wc -l)

if [ "$ENV_USAGE" -gt 0 ]; then
    echo -e "${GREEN}  [통과] 환경 변수 사용 확인 (${ENV_USAGE}개 패턴)${NC}"
else
    echo -e "${YELLOW}  [경고] 환경 변수 사용 패턴 없음 - 확인 필요${NC}"
    ((WARNINGS++))
fi

# ==========================================
# 결과 요약
# ==========================================
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                       검사 결과 요약                        ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $ISSUES_FOUND -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}  모든 검사 통과! 보안 상태 양호합니다.${NC}"
else
    if [ $ISSUES_FOUND -gt 0 ]; then
        echo -e "${RED}  위험 이슈: ${ISSUES_FOUND}개 (즉시 수정 필요!)${NC}"
    fi
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}  경고 사항: ${WARNINGS}개 (검토 권장)${NC}"
    fi
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"

# 종료 코드
if [ $ISSUES_FOUND -gt 0 ]; then
    exit 1
else
    exit 0
fi
