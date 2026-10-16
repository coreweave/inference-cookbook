#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs in the eval Job (CPU only): installs Gym and cwtrain, starts the Gym servers for BENCHMARK,
# collects LIMIT tasks from the hot-load deployment, and prints the scores. For the settings, see
# eval.env.example.
set -euo pipefail
KIT=${KIT_DIR:-/opt/cwtrain-gym}
GYM_DIR=/work/Gym
OUT=/work/results
BENCHMARK=${BENCHMARK:-browsecomp}
apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates >/dev/null
pip install --quiet --no-cache-dir uv
git clone --quiet --branch "${GYM_REF:-v0.6.0}" --depth 1 https://github.com/NVIDIA-NeMo/Gym.git "$GYM_DIR"
cd "$GYM_DIR"
uv sync --quiet --frozen --python 3.13.14
uv pip install --quiet --python "$GYM_DIR/.venv/bin/python" "$KIT"/cwtrain-*.whl
PY="$GYM_DIR/.venv/bin/python"
# The launcher reads its configs next to itself, so run a copy from one directory.
mkdir -p /work/recipe "$OUT"
cp "$KIT"/launch.py "$KIT"/you_search.py "$KIT"/*.yaml "$KIT"/exclude_domains.json /work/recipe/
args=(--gym-dir "$GYM_DIR" --benchmark "$BENCHMARK")
[ "$BENCHMARK" != browsecomp ] || "$PY" /work/recipe/launch.py prepare "${args[@]}"
start=("${args[@]}" --judge-model "${JUDGE_MODEL:-zai-org/GLM-5.2}"
       --judge-base-url "${JUDGE_BASE_URL:-https://api.inference.wandb.ai/v1}"
       --max-context-tokens "${MAX_CONTEXT_TOKENS:-32768}" --max-steps "${MAX_STEPS:-50}")
[ ! -s /secrets/judge/api_key ] || start+=(--judge-api-key-file /secrets/judge/api_key)
[ ! -s /secrets/search/api_key ] || start+=(--search-api-key-file /secrets/search/api_key)
# Own process group, so that stopping it also stops Gym's server processes and Ray.
setsid "$PY" /work/recipe/launch.py start "${start[@]}" > "$OUT/servers.log" 2>&1 &
servers=$!
stop_servers() {
  kill -INT -- "-$servers" 2>/dev/null || return 0
  for _ in $(seq 1 30); do kill -0 "$servers" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$servers" 2>/dev/null || true
}
trap stop_servers EXIT
# Wait for Gym's "All N / N servers ready", or stop if the servers exit first.
for _ in $(seq 1 120); do
  if tr -d '\000' < "$OUT/servers.log" | grep -q "servers ready!"; then break; fi
  kill -0 "$servers" 2>/dev/null || { tail -50 "$OUT/servers.log"; echo "Gym servers exited" >&2; exit 1; }
  sleep 10
done
tr -d '\000' < "$OUT/servers.log" | grep -q "servers ready!" || { tail -50 "$OUT/servers.log"; echo "Gym servers did not start" >&2; exit 1; }
echo "== Gym servers ready for $BENCHMARK"
# The gateway lists the model before its replicas serve, so wait for a completion to succeed.
"$PY" - <<'PY'
import os, time
import httpx
url = os.environ["CWTRAIN_ROLLOUT_BASE_URL"].rstrip("/") + "/chat/completions"
key = open(os.environ["CWTRAIN_ROLLOUT_API_KEY_FILE"]).read().strip()
body = {"model": os.environ["CWTRAIN_ROLLOUT_MODEL"], "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]}
for attempt in range(60):
    try:
        status = httpx.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60).status_code
    except httpx.HTTPError as error:
        status = type(error).__name__
    if status == 200:
        print("== deployment serves", os.environ["CWTRAIN_ROLLOUT_MODEL"], flush=True)
        break
    print(f"deployment not serving yet ({status}); retrying", flush=True)
    time.sleep(10)
else:
    raise SystemExit("the deployment did not serve a completion within 10 minutes")
PY
"$PY" /work/recipe/launch.py collect "${args[@]}" --limit "${LIMIT:-5}" \
  --output "$OUT/coreweave_$BENCHMARK.jsonl"
echo "== scores"
"$PY" - "$OUT/coreweave_$BENCHMARK.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
for row in rows:
    calls = sum(item.get("type") == "function_call" for item in row["response"]["output"])
    print(f"reward={row.get('reward')} tool_calls={calls}")
rewards = [row.get("reward") or 0 for row in rows]
print(f"tasks={len(rows)} mean_reward={sum(rewards) / max(len(rows), 1):.3f}")
PY
