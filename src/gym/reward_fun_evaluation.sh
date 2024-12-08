#!/bin/bash

# Train on different rewards
values=(5 6 7 8 9 10 12 15 18 20 30 40 50 100)

for a in "${values[@]}"; do
    python sb3_solve.py --total_timesteps 4000000 --reward_fun custom --a "$a" --b -1000 --c -2000 &
done
