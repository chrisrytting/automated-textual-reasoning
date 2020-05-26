import os
import pandas as pd
import tensorflow as tf
#import tensorflow_text as text
#import t5

def csv_to_tsv(raw_data_dir, csv_name, target_data_dir, tsv_name):
    df = pd.read_csv(os.path.join(raw_data_dir, csv_name))
    prefixes = []
    targets = []
    max_len = 0
    max_len_tox = 0
    for index, row in df.iterrows():
        prefix = row['question']
        target = row['AnswerKey']
        prefixes.append(prefix)
        targets.append(target)
    with open(os.path.join(target_data_dir, f'{tsv_name}.tsv'), 'w') as f:
        #s1 = text.SentencepieceTokenizer(model='T5/t5/data/test_data/sentencepiece/sentencepiece.model')
        #tokenized_prefix = s1.tokenize([prefix])
        #print(f"Prefix: {prefix} Tokenized Prefix: {tokenized_prefix}")
        print(f"Prefix: {prefix}")
        if len(prefix) > max_len:
            max_len = len(prefix)
#        if tokenized_prefix.shape.as_list()[0] > max_len_tox:
#            max_len_tox = tokenized_prefix.shape


        for prefix, target in zip(prefixes, targets):
            f.write(prefix + '\t' + target + '\n')
    print(f"Max length: {max_len}")

if __name__=="__main__":
    data_dir = 'data/arcprobs/rawdata/ARC-V1-Feb2018-2/ARC-Challenge'
    csv_names = 'ARC-Challenge-Train.csv ARC-Challenge-Dev.csv ARC-Challenge-Test.csv'.split()
    tsv_names = 'arc-train arc-validation arc-test'.split()
    target_data_dir = 'data/arcprobs/arc-challenge'
    for csv_name, tsv_name in zip(csv_names, tsv_names):
        csv_to_tsv(data_dir, csv_name, target_data_dir, tsv_name)


