#!/usr/bin/env bash
# Kaggle pool auth helper for pre_processing_upgrade_7 (KGAT tokens).
# Source this file:  source ops/kaggle_env.sh [account]
# Default account: wagur124705 (matches GitHub wagur1).
# Tokens come from ~/Documents/pool.json — never committed to the repo.

POOL_FILE="${POOL_FILE:-$HOME/Documents/pool.json}"

kaggle_use() {
  local acct="${1:-wagur124705}"
  python3 - "$acct" "$POOL_FILE" <<'PY'
import json, sys
acct, path = sys.argv[1], sys.argv[2]
pool = json.load(open(path))
if acct not in pool:
    sys.exit(f"account '{acct}' not in pool ({', '.join(sorted(pool))})")
print(pool[acct])
PY
}

kaggle_use "${1:-wagur124705}" > /tmp/.kg_token 2>/dev/null || { echo "ERROR: cannot read token" >&2; return 1; }
export KAGGLE_API_TOKEN=$(cat /tmp/.kg_token)
export PATH="$HOME/.local/bin:$PATH"
echo "[kaggle_env] active account: ${1:-wagur124705}"
