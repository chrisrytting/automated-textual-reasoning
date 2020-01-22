import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import test_obj as to

def plot_heatmap_given(accuracies,title, xticklabels=None,yticklabels=None,save=None):
    ax = sns.heatmap(accuracies[::-1,:], 
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels[::-1],
        vmin = 0.0,
        vmax=1.0)
    ax.add_patch(Rectangle((0, 2), 8, 2, fill=False, edgecolor='blue', lw=3))

    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Containers')
    plt.title(title)
    if save:
        plt.savefig(save)
    plt.show()

def substring_accuracy_for_all_experiments(testing=False):
    experiment_names = 'experiment_1 experiment_2 experiment_3 experiment_4'.split()
    other_names = ['', '_0.7_temp', '_0.3_temp', '_0.01_temp']
    n_test_case_list = [20,10,10,10]

    for i in range(4):
        experiment_name=experiment_names[i]
        other = other_names[i]
        n_test_cases = n_test_case_list[i]
        substring_accuracy_for_experiment(experiment_name, n_test_cases, other=other, save='accuracy_fig_{}_{}.png'.format(
            'test' if testing else 'train',experiment_name), testing=testing)

def substring_accuracy_for_experiment(experiment_name,n_test_cases, other = '',save = None,testing=False):
    n_objs_list = np.arange(1,20) 
    n_containers_list = np.arange(2,6) 
    accuracies = to.gather_scores_for_dics(n_objs_list, n_containers_list, experiment_name,n_test_cases,other=other, testing=True)
    plot_heatmap_given(accuracies, experiment_name, xticklabels=n_objs_list,
    yticklabels=n_containers_list, save='substring_acc_{}_{}.png'.format(experiment_name,'test' if testing else 'train'))



def accuracies_heatmap(n_objs_list,n_containers_list,testing=False,save=None):
    accuracies = np.ones((len(n_objs_list),len(n_containers_list))) * -1
    for i,n_objs in enumerate(n_objs_list):
        results_pickle = 'results_dic_{}_nouns_{}_objects_0.01_temp.p'.format(
            'test' if testing else 'train', n_objs
        )
        results_dic = pickle.load(open(results_pickle,'rb'))
        for j,n_containers in enumerate(n_containers_list):

            accuracy = results_dic['{}_objs_{}_containers'.format(n_objs,\
                n_containers)]['score']
            accuracies[i,j] = accuracy
    ax = sns.heatmap(accuracies.T, 
        square=True,
        xticklabels=n_objs_list,
        yticklabels=n_containers_list,
        vmin = 0.0,
        vmax=1.0)
    
    #2,9 2,3
    ax.add_patch(Rectangle((0, 2), 8, 2, fill=False, edgecolor='blue', lw=3))

    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Containers')
    plt.title('Accuracy for temperature = {}'.format(0.01))
    if save:
        plt.savefig(save)


    plt.show()

    
def checkpoint_score_given(
    checkpoint: str,
    n_objs,
    n_containers,
    temperature,
    testing = False):
    result_pickle = 'results_dic_{}_nouns_{}_objects_{}_temp_{}_run_name.p'.format(
        'test' if testing else 'train', n_objs, temperature, checkpoint 
    )
    result_dic = pickle.load(open(result_pickle, 'rb'))
    score = result_dic['{}_objs_{}_containers'.format(n_objs,n_containers)]['score']
    return score

def plot_checkpoint_scores_for(
    checkpoints_list, 
    n_objs,
    n_containers,
    temperature,
    testing = False,
    save = None):
    scores = []
    for checkpoint in checkpoints_list:
        score = checkpoint_score_given(
            checkpoint, n_objs, n_containers, temperature, testing=testing
        )
        scores.append(score)
    plt.plot(scores)
    plt.title('Checkpoint scores for {} objects, {} containers'.format(n_objs,n_containers))
    #plt.xticks(checkpoints_list)
    if save:
        plt.savefig(save)
    plt.show()


            
    

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--testing', action='store_true', help='Include if you want to test on test set nouns')
    parser.add_argument('--save', help='Include this flag and a filename if you want to save the figure')
    parser.add_argument('--plot_checkpoints', action = 'store_true')
    parser.add_argument('--heatmap', action = 'store_true')
    args = parser.parse_args()

    checkpoints_list = ['common_nouns_{}'.format(i) for i in np.arange(9)]

    if args.heatmap:
        accuracies_heatmap(
            np.arange(1,20), 
            np.arange(2,6)[::-1], 
            testing = args.testing,
            save = args.save
        )
    
    if args.plot_checkpoints:
        plot_checkpoint_scores_for(checkpoints_list,
        19,
        2,
        0.1,
        save='trainfig_checkpoints_19_objs_2_containers.png'
        )

        plot_checkpoint_scores_for(checkpoints_list,
        5,
        3,
        0.1,
        save='trainfig_checkpoints_5_objs_3_containers.png'
        )

        plot_checkpoint_scores_for(checkpoints_list,
        19,
        2,
        0.1,
        testing=args.testing,
        save='testfig_checkpoints_19_objs_2_containers.png'
        )

        plot_checkpoint_scores_for(checkpoints_list,
        5,
        3,
        0.1,
        testing=args.testing,
        save='testfig_checkpoints_5_objs_3_containers.png'
        )

