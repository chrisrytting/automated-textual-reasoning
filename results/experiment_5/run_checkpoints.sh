#!/bin/bash

#Round 1 n_objects = 19

#Train set

CUDA_VISIBLE_DEVICES=0  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_0 &
CUDA_VISIBLE_DEVICES=0  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_1 &
CUDA_VISIBLE_DEVICES=1  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_2 &
CUDA_VISIBLE_DEVICES=2  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_3 &
CUDA_VISIBLE_DEVICES=3  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_4 &
CUDA_VISIBLE_DEVICES=4  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_5 &
CUDA_VISIBLE_DEVICES=5  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_6 &
CUDA_VISIBLE_DEVICES=6  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_7 &
CUDA_VISIBLE_DEVICES=7  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_8 &

#Test set

CUDA_VISIBLE_DEVICES=8  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_0 --testing &
CUDA_VISIBLE_DEVICES=8  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_1 --testing &
CUDA_VISIBLE_DEVICES=9  python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_2 --testing &
CUDA_VISIBLE_DEVICES=10 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_3 --testing &
CUDA_VISIBLE_DEVICES=11 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_4 --testing &
CUDA_VISIBLE_DEVICES=12 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_5 --testing &
CUDA_VISIBLE_DEVICES=13 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_6 --testing &
CUDA_VISIBLE_DEVICES=14 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_7 --testing &
CUDA_VISIBLE_DEVICES=15 python3 test_obj.py --temperature 0.1 --n_objects 19 --test_cases 10 --run_name common_nouns_8 --testing &

#Round 2 n_objects = 5

#Train set

CUDA_VISIBLE_DEVICES=1  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_0 &
CUDA_VISIBLE_DEVICES=0  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_1 &
CUDA_VISIBLE_DEVICES=1  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_2 &
CUDA_VISIBLE_DEVICES=2  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_3 &
CUDA_VISIBLE_DEVICES=3  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_4 &
CUDA_VISIBLE_DEVICES=4  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_5 &
CUDA_VISIBLE_DEVICES=5  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_6 &
CUDA_VISIBLE_DEVICES=6  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_7 &
CUDA_VISIBLE_DEVICES=7  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_8 &

#Test set

CUDA_VISIBLE_DEVICES=9  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_0 --testing &
CUDA_VISIBLE_DEVICES=8  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_1 --testing &
CUDA_VISIBLE_DEVICES=9  python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_2 --testing &
CUDA_VISIBLE_DEVICES=10 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_3 --testing &
CUDA_VISIBLE_DEVICES=11 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_4 --testing &
CUDA_VISIBLE_DEVICES=12 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_5 --testing &
CUDA_VISIBLE_DEVICES=13 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_6 --testing &
CUDA_VISIBLE_DEVICES=14 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_7 --testing &
CUDA_VISIBLE_DEVICES=15 python3 test_obj.py --temperature 0.1 --n_objects 5 --test_cases 10 --run_name common_nouns_8 --testing &
wait