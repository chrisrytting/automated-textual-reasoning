import random
from nltk.corpus import wordnet as wn
import os
import pandas as pd

def generate_from_davies():
    davies_dir = 'data/posdata/davies'
    data_dir = 'data/posdata'
    davies_df = pd.read_csv(os.path.join(davies_dir, 'lemmas_60k_words_m1109.txt'), header = 7, sep='\t')

    concreteness_df = pd.read_csv(os.path.join(data_dir,'Concreteness_ratings_Brysbaert_et_al_BRM.csv'))
    concreteness_df = concreteness_df.drop('Bigram Conc.SD Total Percent_known Unknown SUBTLEX'.split(), axis = 1)
    #Change 'Dom_Pos' values of concreteness_df to match 'word' values of davies_df
    left_on_join = []
    right_on_join = []
    df = pd.merge(davies_df, concreteness_df, how='outer', left_on='word', right_on='Word', left_index=False, right_index=False)

    nouns_df = df[df['PoS'] == 'n']
    verbs_df = df[df['PoS'] == 'v']
    
generate_from_davies()


def generate_wn():
    data_dir = 'data/posdata'

    train_containers = set([cont.strip() for cont in open('data/posdata/cont_train_n9.txt','r').readlines()])
    val_containers = set([cont.strip() for cont in open('data/posdata/cont_val_n8.txt','r').readlines()])

    all_nouns = set([n.name()[:-5].replace("_"," ") for n in list(wn.all_synsets('n'))])
    print(f'{len(all_nouns)} nouns in wordnet')
    all_nouns.difference_update(train_containers)
    all_nouns.difference_update(val_containers)
    print(f'{len(all_nouns)} nouns in wordnet after getting rid of container names')

    all_verbs = set([v.name()[:-5].replace("_"," ") for v in list(wn.all_synsets('v'))])
    print(f'{len(all_verbs)} verbs in wordnet')
    all_verbs.difference_update(train_containers)
    all_verbs.difference_update(val_containers)
    print(f'{len(all_verbs)} verbs in wordnet after getting rid of container names')

    all_noun_verbs = set.intersection(all_nouns, all_verbs)
    print(f'{len(all_noun_verbs)} nouns/verbs in wordnet')
    all_nouns.difference_update(all_noun_verbs)
    print(f'{len(all_nouns)} nouns in wordnet after substracting {len(all_noun_verbs)} nouns/verbs from nouns list')
    all_verbs.difference_update(all_noun_verbs)
    print(f'{len(all_verbs)} verbs in wordnet after substracting {len(all_noun_verbs)} nouns/verbs from verbs list')

    n_keep = 2000

    assert len(set.intersection(all_nouns, all_verbs)) == 0
    assert len(set.intersection(all_noun_verbs, all_verbs)) == 0
    assert len(set.intersection(all_nouns, all_noun_verbs)) == 0

    with open(os.path.join(data_dir, 'all_nouns_wn.txt'),'w') as nounf,\
            open(os.path.join(data_dir, 'all_verbs_wn.txt'),'w') as verbf,\
            open(os.path.join(data_dir, 'all_nounverbs_wn.txt'),'w') as nounverbf\
            :
        for noun in all_nouns:
            nounf.write(noun + '\n')
        for verb in all_verbs:
            verbf.write(verb + '\n')
        for nounverb in all_noun_verbs:
            nounverbf.write(nounverb + '\n')

    val_nouns_2000=random.sample(all_nouns, n_keep)
    train_nouns=all_nouns.difference(val_nouns_2000)

    assert len(set.intersection(set(val_nouns_2000), train_nouns)) == 0

    train_nouns_2000=random.sample(train_nouns, n_keep)
    train_nouns_5000=random.sample(train_nouns, 5000)
    val_verbs_2000=random.sample(all_verbs, n_keep)
    val_nounverbs_2000=random.sample(all_noun_verbs, n_keep)

    with open(os.path.join(data_dir, '2000_val_nouns_wn.txt'),'w') as vnounf,\
            open(os.path.join(data_dir, 'all_train_nouns_wn.txt'),'w') as tnounf,\
            open(os.path.join(data_dir, '2000_train_nouns_wn.txt'),'w') as tv2000nounf,\
            open(os.path.join(data_dir, '5000_train_nouns_wn.txt'),'w') as tv5000nounf,\
            open(os.path.join(data_dir, '2000_val_verbs_wn.txt'),'w') as vverbf,\
            open(os.path.join(data_dir, '2000_val_nounverbs_wn.txt'),'w') as vnounverbf:
        for noun in val_nouns_2000:
            vnounf.write(noun + '\n')
        for noun in train_nouns:
            tnounf.write(noun+ '\n')
        for noun in train_nouns_2000:
            tv2000nounf.write(noun+ '\n')
        for noun in train_nouns_5000:
            tv5000nounf.write(noun+ '\n')
        for verb in val_verbs_2000:
            vverbf.write(verb + '\n')
        for nounverb in val_nounverbs_2000:
            vnounverbf.write(nounverb + '\n')

