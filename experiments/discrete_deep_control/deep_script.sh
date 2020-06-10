#!/bin/bash

for value in {0..719}
do
    python3 -O experiments/deep_control/deep_sweep.py $value 50000 200 || exit
done
