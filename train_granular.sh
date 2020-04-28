#!/bin/bash

lrs=(0.003 0.03 0.3 3.0)
bss=(2 4 8)
gpus=(2 3 4 5 6 7 8 9 10 11 12 13)
nlrs=${#lrs[@]}
nbss=${#bss[@]}

for ((iter1=0;iter1<$nbss;iter1++))
do
    bss_ix=$iter1
    for ((iter2=0;iter2<$nlrs;iter2++))
    do
        lrs_ix=$iter2

        #Where it happens
        gpu_id=$((2 + $iter2 + $iter1 * 4))
        CUDA_VISIBLE_DEVICES=$gpu_id python3 t5-train.py --batch_size ${bss[$bss_ix]} --lr ${lrs[$lrs_ix]} --stepsize=100 --nsteps 2500 &

    done
done


wait
