#!/usr/bin/env bash
# Reproduce the channel-2 outputs. Run from the repository root.
#
# Stage A (free, no MFCL runs):  11, 12, 13, 14, 16
# Stage B (evaluation-only MFCL): 15 -> 17     [set C2_RUN_MFCL=1]
# Figures + checks:               18
#
# Every MFCL invocation is evaluation-only (parest_flags(1)=0). No estimation
# run is launched from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
here=analysis/channel2

python3 "$here/11-predictors.py"
python3 "$here/12-bound-transformed.py"
python3 "$here/13-downstream.py"
python3 "$here/14-psi-anchors.py"
python3 "$here/16-report-discrepancy.py"

if [ "${C2_RUN_MFCL:-0}" = "1" ]; then
  python3 "$here/15-profile.py" --groups 17,7 --points 13 --workers "${C2_WORKERS:-4}"
fi
if [ -f out/channel2/profile-objective.csv ]; then
  python3 "$here/17-profile-analysis.py"
else
  echo "skipping profile analysis: no profile-objective.csv (set C2_RUN_MFCL=1)"
fi

python3 "$here/18-figures.py"

cat out/channel2/checks-*.tsv > out/channel2/checks.log
rm -f out/channel2/checks-*.tsv
echo
echo "checks: $(grep -c PASS out/channel2/checks.log) held, $(grep -c FAIL out/channel2/checks.log) did not"
grep FAIL out/channel2/checks.log || true
