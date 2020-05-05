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
            print(f'Doing checkpoint {ckpt}')
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
    print(f'Getting scores for {experiments}')
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

if __name__=="__main__":
    import argparse
    import time
    from multiprocessing import Pool
    import itertools

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--experiments', type=str)
    #parser.add_argument('--checkpoints', type=str)
    #
    #args = parser.parse_args()
    #experiments = args.experiments.split()
    #checkpoints = args.checkpoints.split()
    ##experiments = ([experiments[0]] * len(checkpoints)) + ([experiments[1]] * len(checkpoints))
    ##checkpoints *= 2
    #print(experiments, checkpoints)
    #print(list(itertools.product(experiments, checkpoints)))

    #def twentysecop(n):
    #    mat = np.ones((10000,10000))
    #    mat = mat.dot(mat)
    #    mat *= n
    #    return mat[0,0]
    #args = [1,2,3]
    #with Pool(len(args)) as p:
    #    print(p.map(twentysecop, args))
    #print(twentysecop(1))
    start = time.time()
    experiments = [f'experiment{exp}' for exp in np.arange(9,15)]
    experiment_labels='train val random toverbs verbnouns verbs'.split()
    checkpoints = np.arange(1000000,1001001, 100)
    with Pool() as p:
        #for i in range(len(experiments)):
        args = [([experiment], checkpoints) for experiment in experiments]
        p.starmap(get_scores,args)
        #p.starmap(get_scores,experiments, checkpoints)
        #p.map(get_scores,list(itertools.product(experiments, checkpoints)))
    print(f'Took {time.time() - start} seconds')
    score_dics = get_scores(experiments, checkpoints)
    plot_scores_dic(experiments,experiment_labels,score_dics)





            
        
