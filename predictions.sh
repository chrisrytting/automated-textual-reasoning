#!/bin/bash
checkpoints=($(seq 1000000 100 1002500))
checkpoints={1000000 100 1002500}
experiments=(7 7_ 7_0 7_1)
gpus=($(seq 2 15))
n_checkpoints=${#checkpoints[@]}
n_experiments=${#experiments[@]}
n_gpus=${#gpus[@]}

for ((iter1=0;iter1<${#experiments[@]}; iter1++))
do
    experiment=experiment${experiments[$iter1]}
    for ((iter2=0;iter2<${#checkpoints[@]}; iter2++))
    do
        checkpoint=${checkpoints[$iter2]}
        gpu_id=$((($iter1 * $n_checkpoints + $iter2) % $n_gpus))
        gpu=${gpus[$gpu_id]}

        #Run training script
        CUDA_VISIBLE_DEVICES=$gpu python3 setup_t5_and_predict.py \
            --ckpt $checkpoint\
            --gpu_id 0\
            --experiment $experiment &

        if [ $gpu -eq ${gpus[-1]} ]
        then
            echo waiting at \
                gpu $gpu \
                experiment $experiment \
                checkpoint $checkpoint
            wait
        fi

    done

#Will I lose this?
done



#Experiment 1

#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1000000 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1000250 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1000500 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1000750 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1001000 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt  1001250 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt  1001500 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1001750 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt 1002000 --gpu_id  0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt 1002250 --gpu_id  0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt 1002500 --gpu_id  0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1002750 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt  1003000 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt  1003250 --gpu_id 0 --experiment 'experiment1' &
#
#wait
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1003500 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1003750 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1004000 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1004250 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1004500 --gpu_id 0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt 1004750 --gpu_id  0 --experiment 'experiment1' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt 1005000 --gpu_id  0 --experiment 'experiment1' &
#                     
##Experiment2
#
#CUDA_VISIBLE_DEVICES=9   python3 setup_t5_and_predict.py --ckpt  1000000 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=10  python3 setup_t5_and_predict.py --ckpt  1000250 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=11  python3 setup_t5_and_predict.py --ckpt  1000500 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=12  python3 setup_t5_and_predict.py --ckpt  1000750 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=13  python3 setup_t5_and_predict.py --ckpt  1001000 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=14  python3 setup_t5_and_predict.py --ckpt  1001250 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt  1001500 --gpu_id 0 --experiment 'experiment2' &
#
#wait
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1001750 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt 1002000 --gpu_id  0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt 1002250 --gpu_id  0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt 1002500 --gpu_id  0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1002750 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt  1003000 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt  1003250 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1003500 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt  1003750 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt  1004000 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt  1004250 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1004500 --gpu_id 0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt 1004750 --gpu_id  0 --experiment 'experiment2' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt 1005000 --gpu_id  0 --experiment 'experiment2' &
#
#wait
#
##Experiment 4
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1000000 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1000250 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1000500 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1000750 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1001000 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt  1001250 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt  1001500 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1001750 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt 1002000 --gpu_id  0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt 1002250 --gpu_id  0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt 1002500 --gpu_id  0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1002750 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt  1003000 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt  1003250 --gpu_id 0 --experiment 'experiment4' &
#
#wait
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1003500 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1003750 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1004000 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1004250 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1004500 --gpu_id 0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt 1004750 --gpu_id  0 --experiment 'experiment4' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt 1005000 --gpu_id  0 --experiment 'experiment4' &
#
##Experiment 5
#                     
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1000000 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt  1000250 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt  1000500 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt  1000750 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1001000 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt  1001250 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt  1001500 --gpu_id 0 --experiment 'experiment5' &
#
#wait
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1001750 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt 1002000 --gpu_id  0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt 1002250 --gpu_id  0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt 1002500 --gpu_id  0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1002750 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt  1003000 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt  1003250 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1003500 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt  1003750 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt  1004000 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt  1004250 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1004500 --gpu_id 0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt 1004750 --gpu_id  0 --experiment 'experiment5' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt 1005000 --gpu_id  0 --experiment 'experiment5' &
#
#wait
#
##Experiment 6
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1000000 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1000250 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1000500 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1000750 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1001000 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt  1001250 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt  1001500 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=9  python3 setup_t5_and_predict.py --ckpt  1001750 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=10 python3 setup_t5_and_predict.py --ckpt 1002000 --gpu_id  0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=11 python3 setup_t5_and_predict.py --ckpt 1002250 --gpu_id  0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=12 python3 setup_t5_and_predict.py --ckpt 1002500 --gpu_id  0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=13 python3 setup_t5_and_predict.py --ckpt  1002750 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=14 python3 setup_t5_and_predict.py --ckpt  1003000 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=15 python3 setup_t5_and_predict.py --ckpt  1003250 --gpu_id 0 --experiment 'experiment6' &
#
#wait
#
#CUDA_VISIBLE_DEVICES=2  python3 setup_t5_and_predict.py --ckpt  1003500 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=3  python3 setup_t5_and_predict.py --ckpt  1003750 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=4  python3 setup_t5_and_predict.py --ckpt  1004000 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=5  python3 setup_t5_and_predict.py --ckpt  1004250 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=6  python3 setup_t5_and_predict.py --ckpt  1004500 --gpu_id 0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=7  python3 setup_t5_and_predict.py --ckpt 1004750 --gpu_id  0 --experiment 'experiment6' &
#CUDA_VISIBLE_DEVICES=8  python3 setup_t5_and_predict.py --ckpt 1005000 --gpu_id  0 --experiment 'experiment6' &
