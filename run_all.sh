#!/bin/bash
# Runs the seven steps in order, then the checks. Needs Python 3.9+ and PyTorch (pip install torch). CPU is enough.
# Steps 1 to 6 take about a minute or two each on a laptop. Step 7 trains nine agents and takes about 15 minutes.
set -e
mkdir -p runs
for step in step1_full step2_window step3_hybrid step4_experts step5_draft step6_grpo step7_redistribute; do
  echo "== $step"
  python3 $step.py
  echo
done
python3 tests.py
