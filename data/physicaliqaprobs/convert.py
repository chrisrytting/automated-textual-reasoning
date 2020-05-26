import os
import pandas as pd
import tensorflow as tf
import jsonlines

#import tensorflow_text as text
#import t5

def unload_jsonl(raw_data_dir, filename):
    objs = []
    with jsonlines.open(os.path.join(raw_data_dir, filename)) as reader:
        for obj in reader:
            objs.append(obj)
    return objs

def generate_lines(labels, objs):
    lines = []
    for label, obj in zip(labels, objs):
        line = f"(goal) {obj['goal']} (0) {obj['sol1']} (1) {obj['sol2']}\t{label}"
        lines.append(line)
    return lines

def write_lines_to_tsv(lines, target_data_dir, tsv_name):
    with open(os.path.join(target_data_dir, tsv_name),'w') as f:
        for line in lines:
            f.write(line + '\n')


def csv_to_tsv(raw_data_dir, target_data_dir):

    devlabels = [int(label.strip()) for label in open(os.path.join(raw_data_dir, 'dev-labels.lst'), 'r').readlines()]
    trainlabels = [int(label.strip()) for label in open(os.path.join(raw_data_dir, 'train-labels.lst'), 'r').readlines()]
    
    devobjs = unload_jsonl(raw_data_dir, 'dev.jsonl')
    trainobjs = unload_jsonl(raw_data_dir, 'train.jsonl')

    trainlines = generate_lines(trainlabels, trainobjs)
    devlines = generate_lines(devlabels, devobjs)

    write_lines_to_tsv(trainlines, target_data_dir, 'atr-train.tsv')
    write_lines_to_tsv(devlines, target_data_dir, 'atr-validation.tsv')


if __name__=="__main__":
    data_dir = 'data/physicaliqaprobs/rawdata/physicaliqa-train-dev'
    target_data_dir = 'data/physicaliqaprobs/'
    csv_to_tsv(data_dir, target_data_dir)


