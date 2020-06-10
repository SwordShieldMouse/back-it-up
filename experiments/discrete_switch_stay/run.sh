#!/bin/bash
for value in {0..47}
do
    python3 experiments/switch-stay/run.py $value || exit
done