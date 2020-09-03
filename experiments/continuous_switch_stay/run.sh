#!/bin/bash
for value in {0..29}
do
    for std in {0..0}
    do
        python3 experiments/continuous-switch-stay/run.py $value $std || exit
    done
done