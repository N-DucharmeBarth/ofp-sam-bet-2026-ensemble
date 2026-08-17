#!/usr/bin/env bash
# Reproduce every out/ artefact for the reporting-rate penalty analysis.
# Run from the repository root. Task 3 gates the rest: 01 exits non-zero if the
# penalty reconstruction does not match the reported objective component.
set -euo pipefail

cd "$(dirname "$0")/../.."
here=analysis/rr-penalty

python3 "$here/01-groups-and-reconstruction.py"
python3 "$here/02-marginal-effect.py"
python3 "$here/03-excursion.py"
python3 "$here/04-mixing.py"
python3 "$here/05-psi.py"
python3 "$here/06-bound-floor.py"
python3 "$here/07-figures.py"

cat out/checks-0*.txt > out/checks.tsv
rm -f out/checks-0*.txt
echo
echo "checks: $(grep -c PASS out/checks.tsv) passed, $(grep -c FAIL out/checks.tsv) failed"
grep FAIL out/checks.tsv || true
