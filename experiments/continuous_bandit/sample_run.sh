#!/bin/bash
for value in {0..29}
do
    python3 experiments/continuous_bandit/sample_run.py $value || exit
done