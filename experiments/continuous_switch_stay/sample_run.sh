#!/bin/bash
for value in {0..19}
do
    for std in {0..0}
    do
        python3 experiments/continuous-switch-stay/sample_run.py $value $std || exit
    done
done