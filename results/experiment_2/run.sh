#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 test_obj.py  --temperature 0.7 --n_objects 1 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=1 python3 test_obj.py  --temperature 0.7 --n_objects 2 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=2 python3 test_obj.py  --temperature 0.7 --n_objects 3 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=3 python3 test_obj.py  --temperature 0.7 --n_objects 4 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=4 python3 test_obj.py  --temperature 0.7 --n_objects 5 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=5 python3 test_obj.py  --temperature 0.7 --n_objects 6 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=6 python3 test_obj.py  --temperature 0.7 --n_objects 7 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=7 python3 test_obj.py  --temperature 0.7 --n_objects 8 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=8 python3 test_obj.py  --temperature 0.7 --n_objects 9 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=9 python3 test_obj.py  --temperature 0.7 --n_objects 10 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=10 python3 test_obj.py --temperature 0.7 --n_objects 11 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=11 python3 test_obj.py --temperature 0.7 --n_objects 12 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=12 python3 test_obj.py --temperature 0.7 --n_objects 13 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=13 python3 test_obj.py --temperature 0.7 --n_objects 14 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=14 python3 test_obj.py --temperature 0.7 --n_objects 15 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=14 python3 test_obj.py --temperature 0.7 --n_objects 16 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=15 python3 test_obj.py --temperature 0.7 --n_objects 17 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=15 python3 test_obj.py --temperature 0.7 --n_objects 18 --test_cases 10 --run_name common_nouns &
CUDA_VISIBLE_DEVICES=3 python3 test_obj.py --temperature 0.7 --n_objects 19 --test_cases 10 --run_name common_nouns &

#Testings

CUDA_VISIBLE_DEVICES=0 python3 test_obj.py  --temperature 0.7 --n_objects 1 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=0 python3 test_obj.py  --temperature 0.7 --n_objects 2 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=1 python3 test_obj.py  --temperature 0.7 --n_objects 3 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=1 python3 test_obj.py  --temperature 0.7 --n_objects 4 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=2 python3 test_obj.py  --temperature 0.7 --n_objects 5 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=3 python3 test_obj.py  --temperature 0.7 --n_objects 6 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=4 python3 test_obj.py  --temperature 0.7 --n_objects 7 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=5 python3 test_obj.py  --temperature 0.7 --n_objects 8 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=6 python3 test_obj.py  --temperature 0.7 --n_objects 9 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=7 python3 test_obj.py  --temperature 0.7 --n_objects 10 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=8 python3 test_obj.py  --temperature 0.7 --n_objects 11 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=9 python3 test_obj.py  --temperature 0.7 --n_objects 12 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=10 python3 test_obj.py --temperature 0.7 --n_objects 13 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=11 python3 test_obj.py --temperature 0.7 --n_objects 14 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=12 python3 test_obj.py --temperature 0.7 --n_objects 15 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=13 python3 test_obj.py --temperature 0.7 --n_objects 16 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=14 python3 test_obj.py --temperature 0.7 --n_objects 17 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=15 python3 test_obj.py --temperature 0.7 --n_objects 18 --test_cases 10 --run_name common_nouns --testing &
CUDA_VISIBLE_DEVICES=2 python3 test_obj.py --temperature 0.7 --n_objects 19 --test_cases 10 --run_name common_nouns --testing &

wait