#!/bin/bash
for value in {0..59}
do
    python3 ./experiments/discrete_bandit/run.py $mode $value || exit
done