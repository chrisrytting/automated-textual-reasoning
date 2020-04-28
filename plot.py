import testing
import os
import numpy as np
import json
from matplotlib import pyplot as plt


def score_and_plot(experiment):
    checkpoints = np.arange(1000000,1005001,250)
    bleus, exacts, substrs = [],[],[]
    score_dics = {}
    json_path = f'experiments/{experiment}/testing/score_dics.json'
    if not os.path.exists(json_path):
        for ckpt in checkpoints:
            score_dic = testing.compare_txt_files(f'experiments/{experiment}/testing/targets.txt', f'experiments/{experiment}/testing/predictions.txt-{ckpt}', np.arange(2,20), np.arange(2,6))
            bleus.append(score_dic['avg_bleu_all'])
            exacts.append(score_dic['avg_exact_all'])
            substrs.append(score_dic['avg_substr_all'])
            score_dics[str(ckpt)] = score_dic

        with open(json_path,'w') as outfile:
            #score_dics_string = json.dumps(score_dics)
            #json.dumps(score_dics_string,outfile)
            json.dump(score_dics,outfile)
    else:
        score_dics = json.load(open(json_path,'r'))
        for ckpt in checkpoints:
            score_dic = score_dics[str(ckpt)]
            bleus.append(score_dic['avg_bleu_all'])
            exacts.append(score_dic['avg_exact_all'])
            substrs.append(score_dic['avg_substr_all'])


    plt.close()
    plt.plot(bleus, label='bleus')
    plt.plot(exacts, label='exacts')
    plt.plot(substrs, label='substrs')
    plt.legend()
    plt.title('Accuracy over training')
    plt.xlabel('Checkpoints')
    plt.ylim(0,1)
    plt.ylabel('Accuracy')
    plt.savefig(f'experiments/{experiment}/testing/accuracies.png')
    plt.show()





