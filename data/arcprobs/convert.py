import os
import pandas as pd

def csv_to_tsv(raw_data_dir, csv_name, target_data_dir, tsv_name=None):
    df = pd.read_csv(os.path.join(raw_data_dir, csv_name))
    prefixes = []
    targets = []
    for instance in traindf:
        prefix = instance['question']
        target = instance['AnswerKey']
        prefixes.append(prefix)
        targets.append(target)
    with open(os.path.join(target_data_dir, '-'.join('prefixes',tsv_name) + '.tsv'), 'w') as pfile,
        open(os.path.join(target_data_dir, '-'.join('targets',tsv_name) + '.tsv'), 'w') as tfile:
             for prefix, target in zip(prefixes, targets):
                 pfile.write(prefix + '\n')
                 tfile.write(target + '\n')

if __name__=="__main__":
data_dir = 'data/arcprobs/rawdata/ARC-V1-Feb2018-2/ARC-Easy'
file_name = 'ARC-Easy-Train.csv'
csv_to_tsv(
