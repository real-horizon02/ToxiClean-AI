#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - openenv-core: pip install openenv-core
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x scripts/validate-submission.sh
#     ./scripts/validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./scripts/validate-submission.sh https://my-team.hf.space
#   ./scripts/validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}  ToxiClean AI${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ===========================================================================
# Step 1/5: Ping HF Space /reset endpoint
# ===========================================================================
log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset with HTTP 200"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Your Space must use sdk: docker and expose the FastAPI server."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

# ===========================================================================
# Step 2/5: Validate mandatory environment variables in .env.example
# ===========================================================================
log "${BOLD}Step 2/5: Checking mandatory env vars in .env.example${NC} ..."

ENV_EXAMPLE="$REPO_DIR/.env.example"
MISSING_VARS=()

if [ ! -f "$ENV_EXAMPLE" ]; then
  fail ".env.example not found in repo root"
  stop_at "Step 2"
fi

for VAR in API_BASE_URL MODEL_NAME HF_TOKEN; do
  if ! grep -q "^${VAR}=" "$ENV_EXAMPLE" && ! grep -q "^#.*${VAR}=" "$ENV_EXAMPLE" && ! grep -q "${VAR}" "$ENV_EXAMPLE"; then
    MISSING_VARS+=("$VAR")
  fi
done

if [ ${#MISSING_VARS[@]} -eq 0 ]; then
  pass "All mandatory env vars documented in .env.example: API_BASE_URL, MODEL_NAME, HF_TOKEN"
else
  fail "Missing env vars in .env.example: ${MISSING_VARS[*]}"
  hint "Add the following to .env.example: ${MISSING_VARS[*]}"
  stop_at "Step 2"
fi

# ===========================================================================
# Step 3/5: Check inference.py exists at repo root with correct log format
# ===========================================================================
log "${BOLD}Step 3/5: Validating inference.py${NC} ..."

INFERENCE="$REPO_DIR/inference.py"

if [ ! -f "$INFERENCE" ]; then
  fail "inference.py not found at repo root"
  hint "The inference script MUST be named inference.py and live in the repo root."
  stop_at "Step 3"
fi

# Check it uses the OpenAI client
if ! grep -q "from openai import OpenAI\|import openai" "$INFERENCE"; then
  fail "inference.py does not import the OpenAI client"
  hint "Participants MUST use the OpenAI client for all LLM calls."
  hint "Add: from openai import OpenAI"
  stop_at "Step 3"
fi

# Check mandatory log format tokens
MISSING_LOGS=()
grep -q "\[START\]" "$INFERENCE" || MISSING_LOGS+=("[START]")
grep -q "\[STEP\]"  "$INFERENCE" || MISSING_LOGS+=("[STEP]")
grep -q "\[END\]"   "$INFERENCE" || MISSING_LOGS+=("[END]")

if [ ${#MISSING_LOGS[@]} -eq 0 ]; then
  pass "inference.py present at root, uses OpenAI client, emits [START]/[STEP]/[END] logs"
else
  fail "inference.py is missing mandatory log markers: ${MISSING_LOGS[*]}"
  hint "Your inference script MUST emit [START], [STEP], and [END] lines to stdout."
  stop_at "Step 3"
fi

# ===========================================================================
# Step 4/5: Docker build
# ===========================================================================
log "${BOLD}Step 4/5: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 4"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 4"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 4"
fi

# ===========================================================================
# Step 5/5: openenv validate
# ===========================================================================
log "${BOLD}Step 5/5: Running openenv validate${NC} ..."

if ! command -v openenv &>/dev/null; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 5"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 5"
fi

# ===========================================================================
# Summary
# ===========================================================================
printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 5/5 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
