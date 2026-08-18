#!/bin/bash
# Orchestrates Experiments A and B (paired-refit-decision.md) plus the
# representative-member fixed-reporting-rate runs.
#
# Stage 1 (10 runs, independent): Experiment A's 8 cells (2 arms x tau{1.2,1.4}
#   x K{0.10,0.30}, h/M0/creep at ensemble median) plus the 2 median-*-fixedRR
#   runs (both arms, priored PS groups fixed at their tag-seeding prior mean).
# Between stages: fill Experiment B's 4 expB-C-* rows with the fitted X-hat
#   each cell's Experiment A include run actually estimated.
# Stage 2 (4 runs, depends on stage 1): Experiment B's exclude-arm runs with
#   the priored groups fixed at that fitted X-hat (channel-1-only removal).
#
# Concurrency capped at CONCURRENCY (default 2). Each run is single-threaded
# (the ensemble-001 benchmark: ~7h wall clock, ~98% single-core), so this
# machine's 4 cores could in principle run 4 at once, but an actual 4-way
# attempt drove memory from ~5GB to ~14GB used (of 15GB, no swap) within two
# minutes of entering Phase 1 and one run was killed. 2 keeps per-run memory
# (observed ~3.5GB early in Phase 1, ceiling unmeasured) comfortably inside
# budget; raise only with headroom actually confirmed by watching `free -h`
# through a full run at the higher setting.
set -uo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../../.." && pwd)
LOGDIR="$REPO/out/channel2/experiments"
mkdir -p "$LOGDIR"
CONCURRENCY=${CONCURRENCY:-2}

STAGE1=(
  expA-inc-tau1.2-K0.1 expA-exc-tau1.2-K0.1
  expA-inc-tau1.2-K0.3 expA-exc-tau1.2-K0.3
  expA-inc-tau1.4-K0.1 expA-exc-tau1.4-K0.1
  expA-inc-tau1.4-K0.3 expA-exc-tau1.4-K0.3
  median-inc-fixedRR median-exc-fixedRR
)
STAGE2=(
  expB-C-tau1.2-K0.1 expB-C-tau1.2-K0.3
  expB-C-tau1.4-K0.1 expB-C-tau1.4-K0.3
)

run_one() {
  local id=$1
  local log="$LOGDIR/${id}.log"
  local rc
  {
    echo "START $id: $(date -u +%FT%TZ)"
    time "$HERE/../run-experiment" "$id"
    rc=$?
    echo "END $id: $(date -u +%FT%TZ) rc=$rc"
  } > "$log" 2>&1
  echo "$id exit=$rc $(date -u +%FT%TZ)" >> "$LOGDIR/status.log"
}

run_batch() {
  local -a ids=("$@")
  local running=0
  for id in "${ids[@]}"; do
    run_one "$id" &
    running=$((running + 1))
    if [ "$running" -ge "$CONCURRENCY" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

# Lightweight memory watchdog: if used memory ever exceeds 90% of total with
# no swap, log it loudly so a later check-in catches it even if nothing
# actually OOM-kills. Not cleaned up on abrupt kill of this script -- it is a
# harmless orphaned loop (one `free` call every 2 minutes) if that happens.
(
  while true; do
    read -r _ total used < <(free -m | awk '/^Mem:/{print $1,$2,$3}')
    pct=$((used * 100 / total))
    echo "$(date -u +%FT%TZ) mem_used_pct=$pct used=${used}MB total=${total}MB" >> "$LOGDIR/memory.log"
    if [ "$pct" -ge 90 ]; then
      echo "$(date -u +%FT%TZ) WARNING: memory at ${pct}%" >> "$LOGDIR/status.log"
    fi
    sleep 120
  done
) &
MEMWATCH_PID=$!

echo "=== Stage 1: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"
run_batch "${STAGE1[@]}"
echo "=== Stage 1 complete: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"

echo "=== Filling Experiment B fitted rates: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"
python3 "$HERE/fill-expB-rates.py" 2>&1 | tee -a "$LOGDIR/status.log"

echo "=== Stage 2: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"
run_batch "${STAGE2[@]}"
echo "=== Stage 2 complete: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"

echo "=== ALL DONE: $(date -u +%FT%TZ) ===" | tee -a "$LOGDIR/status.log"
n_ok=$(grep -c "exit=0" "$LOGDIR/status.log" || true)
echo "successful runs: $n_ok"
grep "exit=" "$LOGDIR/status.log" | grep -v "exit=0" > "$LOGDIR/failures.log" || true
n_fail=$(wc -l < "$LOGDIR/failures.log")
echo "failed runs: $n_fail (see failures.log)"
kill "$MEMWATCH_PID" 2>/dev/null || true
