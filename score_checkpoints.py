from testing import compare_txt_files
import seaborn
import os
import numpy as np
import json
from matplotlib import pyplot as plt




def score_experiment(
    checkpoints,
    experiment,
    json_file_name="score_dics.json",
):
    """
    Calls compare_txt_files for targets.txt and predictions corresponding to 
    each ckpt in checkpoints. Inserts the bleu scores, exact scores, and substr
    scores into a dictionary to be saved as a json in the experiment directory.

    Arguments:
    
    checkpoints {list} -- The checkpoints one is interested in scores for. 
    experiment {str} -- The experiment one wants to score.
    json_file_name {str} -- The name of the file to save the score to. 

    Returns:

    The score_dics, whose keys are the checkpoints in string form.

    """
    bleus, exacts, substrs = [], [], []
    score_dics = {}
    json_path = f"experiments/{experiment}/testing/{json_file_name}"
    if not os.path.exists(json_path):
        for ckpt in checkpoints:
            score_dic = compare_txt_files(
                f"experiments/{experiment}/testing/targets.txt",
                f"experiments/{experiment}/testing/predictions.txt-{ckpt}",
                np.arange(2, 20),
                np.arange(2, 6),
            )
            bleus.append(score_dic["avg_bleu_all"])
            exacts.append(score_dic["avg_exact_all"])
            substrs.append(score_dic["avg_substr_all"])
            score_dics[str(ckpt)] = score_dic
        with open(json_path, "w") as outfile:
            json.dump(score_dics, outfile)
    else:
        score_dics=json.load(open(json_path,'r'))
    return score_dics
        
def extract_series(checkpoints, score_dics):
    """
    Looks across the score_dics of a given experiment for the checkpoints 
    indicated and extracts the series of scores for each metric for each 
    of the checkpoints in order.

    Arguments:

    checkpoints {list} -- The checkpoints one is interested in scores for. These
                          are the keys of score_dics.
    score_dics {dic} -- The score_dics for an experiment.

    Returns:

    A dictionary whose keys indicate which metric's scores are housed in the
    value.
    """

    keys = 'bleu exact substr'.split()
    score_dic_keys = 'avg_bleu_all avg_exact_all avg_substr_all'.split()
    series_dic= {key: [] for key in keys}
    key_dic = {key:score_dic_key for key,score_dic_key in zip(keys, score_dic_keys)}

    for ckpt in checkpoints:
        for key in keys:
            series_dic[key].append(score_dics[str(ckpt)][key_dic[key]])
    return series_dic


def get_scores(experiments, checkpoints):
    """
    Returns a dictionary whose values are sequences of metric scores for each
    experiment and whose keys are the names of the metric scores
    """
    keys = 'bleu exact substr'.split()
    #keys = 'avg_bleu_all avg_exact_all avg_substr_all'.split()
    scores_dic = {key : [] for key in keys}
    for experiment in experiments:
        score_dics = score_experiment(checkpoints, experiment)
        series_dic = extract_series(checkpoints, score_dics)
        for key in keys:
            scores_dic[key].append(series_dic[key])
    return scores_dic

def plot_scores_dic(experiments, experiment_labels, scores_dic):
    keys = list(scores_dic.keys())
    for key in keys:
        plt.close()
        for ix,experiment_label in enumerate(experiment_labels):
            plt.plot(scores_dic[key][ix], label = experiment_label)
        plt.legend()
        plt.title(f'{key}')
        plt.xlabel('Checkpoints')
        plt.ylim(0,1)
        plt.ylabel('Score')
        plt.savefig(f'experiments/{experiments[ix]}/testing/accuracies_{key}.png')

experiment_dirs = 'experiment1.experiment2.experiment4.experiment5.experiment6'.split('.')
experiments = 'to verbs.train.random.validation.verbs'.split('.')



            
        
