#!/bin/bash

for i in {1..5}
do
    seed=$((21 * i))
    echo "Running with seed $seed"
    python cli.py --seed $seed
done