#!/bin/bash
for value in {0..29}
do
    python3 experiments/continuous_bandit/run.py $value || exit
done