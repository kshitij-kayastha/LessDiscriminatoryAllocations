#!/usr/bin/env bash
# Run all nine true-needs configurations in parallel, one process each.
#
# Each process is single-threaded pure Python and peaks around 0.25 GB, so nine
# at once fit comfortably. Launch order follows the priority Kshitij gave, which
# only matters if you cut the run short.
#
#   bash scripts/run_all.sh
#   tail -f logs/run_*.log
set -u
cd "$(dirname "$0")/.."
mkdir -p logs data/results/allocations
PY=.venv/bin/python

launch () {
  local rate=$1 qual=$2
  local name="rate-${rate}_${qual}"
  # --seed=-1 leaves NumPy unseeded, matching the notebook's cohort sampling.
  OMP_NUM_THREADS=1 nohup "$PY" -I scripts/run_experiment.py \
      --selection-rate "$rate" --qualification-column "$qual" \
      --seed=-1 --skip-existing \
      > "logs/run_${name}.log" 2>&1 &
  echo "launched ${name} (pid $!)"
}

# Longest first so they finish together.
launch 0.5  qualified_gagne_1
launch 0.5  qualified_gagne_2
launch 0.5  qualified_gagne_3
launch 0.25 qualified_gagne_1
launch 0.25 qualified_gagne_2
launch 0.25 qualified_gagne_3
launch 0.1  qualified_gagne_1
launch 0.1  qualified_gagne_2
launch 0.1  qualified_gagne_3

echo
echo "All nine launched. Watch with:  tail -f logs/run_*.log"
echo "Expected wall clock: ~18-20 h (the three rate-0.5 runs dominate)."
wait
echo "ALL RUNS COMPLETE"
