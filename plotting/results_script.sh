#!/bin/bash
for value in {0..7}
do
    python3 ./plotting/kl_results.py 0.5 $value || exit
done
