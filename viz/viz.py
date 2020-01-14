import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def accuracies_heatmap(results_pickle,n_objs_list,n_containers_list):
    results_dic = pickle.load(open(results_pickle,'rb'))
    accuracies = np.ones((len(n_objs_list),len(n_containers_list))) * -1
    for i,n_objs in enumerate(n_objs_list):
        for j,n_containers in enumerate(n_containers_list):
            accuracy = results_dic['{}_objs_{}_containers'.format(n_objs,\
                n_containers)]['score']
            accuracies[i,j] = accuracy
    ax = sns.heatmap(accuracies.T, 
        square=True,
        xticklabels=n_objs_list,
        yticklabels=n_containers_list)
    
    #2,9 2,3
    ax.add_patch(Rectangle((0, 0), 8, 2, fill=False, edgecolor='blue', lw=3))

    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Containers')
    plt.title('Accuracy')
    plt.show()

    
            
    

if __name__=="__main__":
    accuracies_heatmap('results_dic.p', np.arange(1,15), np.arange(2,6)[::-1])


