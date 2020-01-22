import generate_templates as gt
import re
import pickle
import numpy as np

def conduct_tests(
    n_objs_to_test, 
    n_containers_to_test, 
    sess, 
    gpt2, 
    run_name,
    scenario_type,
    testing = False,
    test_cases = 20,
    temperature = 0.1):
    """Find accuracies on a test set for several n_objs and n_containers
    
    Arguments:
        n_objs_to_test {list{ints}} -- list of different n_objs wanting to test for
        n_containers_to_test {list{ints}} -- list of different n_containers wanting to test for
        
    
    Returns:
        np.array -- accuracies for each of the n_objs on axis 0 and n_containers on axis 1
    """
    results_dic = {}
    for i, n_objs in enumerate(n_objs_to_test):
        print('Iteration {}'.format(i))
        for j, n_containers in enumerate(n_containers_to_test):
            result_dic = conduct_test(n_objs,n_containers,sess, gpt2, run_name, \
                scenario_type,test_cases=test_cases,testing=testing, temperature = temperature)
            print('Score for {}_objs_{}_containers_{}_checkpoint = {}'.format(n_objs,n_containers,result_dic['score']))
            results_dic['{}_objs_{}_containers'.format(n_objs,n_containers)] = result_dic
    return results_dic

def conduct_test(n_objs, n_containers, sess, gpt2, run_name, scenario_type, \
    test_cases=10, testing=False,temperature = 0.7):
    truncate = '<END>'
    acc_count = 0.0
    result_dic = {}
    for i in range(test_cases):
        true_scenario = gt.generate_scenario(n_objs, n_containers, \
            scenario_type,test=testing)
        prefix = re.search('.*Took[^\.]*', true_scenario).group(0)
        predicted_scenario = gpt2.generate(sess, prefix = prefix, \
            run_name=run_name, truncate =truncate,return_as_list=True,\
                temperature = temperature)[0] + truncate
        match = true_scenario == predicted_scenario

        #Log results in a dic
        result_dic['true_scenario_{}'.format(i)] = true_scenario
        result_dic['prefix_{}'.format(i)] = prefix
        result_dic['predicted_scenario_{}'.format(i)] = predicted_scenario
        result_dic['match_{}'.format(i)] = match
        if match:
            acc_count += 1
    score = acc_count / test_cases
    result_dic['score'] = score
    return result_dic

#Realized this wont work because you have to start a new python session every time you make a new model.
def conduct_test_across_checkpoints(checkpoint_list,n_objs, n_containers,sess,gpt2,
    scenario_type, test_cases = 10, testing = False, temperature = 0.7):
    results_dic = {}
    for checkpoint in checkpoint_list:
        result_dic = conduct_test(n_objs,n_containers,sess,gpt2,checkpoint,
            scenario_type,test_cases=test_cases,testing =testing, temperature=temperature)
        print('Score for {}_objs_{}_containers at checkpoint {} = {}'.format(
            n_objs, n_containers, checkpoint, result_dic['score']))
        results_dic['{}_objs_{}_containers_{}_checkpoint'.format(n_objs,n_containers,checkpoint)] = result_dic
    return results_dic

def score_dic_on_substrings(result_dic, n_containers, n_test_cases = 20):
    #There are multiple containers in each result_dic
        case_scores = []
        for i in range(n_test_cases):
            true_scenario = result_dic['true_scenario_{}'.format(i)]
            prefix = result_dic['prefix_{}'.format(i)]
            generated_scenario = result_dic['predicted_scenario_{}'.format(i)]
            true_fs = true_scenario.replace(prefix,'')
            true_fs_components = true_fs.split('.')[1:-1]
            generated_fs = generated_scenario.replace(prefix,'')
            score = 0.0
            for true_fs_component in true_fs_components:
                if true_fs_component in generated_fs:
                #    print(true_fs_component, ' is inside')
                    score += 1
                #else:
                #    print(true_fs_component, ' is outside')
       	    score /= len(true_fs_components)
            case_scores.append(score)
            case_score_mean = np.mean(case_scores)
        return case_score_mean

def gather_scores_for_dics(n_objs_list, n_containers_list, experiment_name, n_test_cases, other='', testing = False):
    accuracies = np.ones((len(n_containers_list), len(n_objs_list))) * -1
    for i,n_objs in enumerate(n_objs_list):
        pickle_name = 'results/{}/results_dic_{}_nouns_{}_objects{}.p'.format(experiment_name,'test' if testing else 'train', n_objs,other)
        results_dic = load_pickle(pickle_name)
        for j,n_containers in enumerate(n_containers_list):
            key_name = '{}_objs_{}_containers'.format(n_objs,n_containers)
            result_dic = results_dic[key_name]
            score = score_dic_on_substrings(result_dic, n_containers,n_test_cases=n_test_cases)
            #print(score)
            accuracies[j,i] = score
    #print(accuracies)
    return accuracies


def score_pickle(pickle_name):
    result_dic = load_pickle(pickle_name)
    score = score_dic_on_substrings(result_dic, 19)
    return score

def load_pickle(pickle_name):
    """Loads a pickle"""
    result = pickle.load(open(pickle_name, 'rb'))
    return result

def dump_pickle(thing, pickle_name):
    """Dumps thing into pickle_name"""
    return pickle.dump(thing, open(pickle_name, 'wb'))
        



        
        



if __name__=="__main__":
    import gpt_2_simple as gpt2
    import time
    import numpy as np
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Run gpt2 test suite')
    parser.add_argument('--n_objects', type=int)
    #parser.add_argument('--n_containers', type=int)
    parser.add_argument('--test_cases', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--run_name')
    parser.add_argument('--testing', action='store_true')
    args = parser.parse_args()
    
    n_objects = [args.n_objects]
    run_name = args.run_name
    testing = args.testing
    test_cases = args.test_cases
    temperature = args.temperature



    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess,run_name=run_name)
    start = time.time()
    results_dic = conduct_tests(n_objects, np.arange(2,6),sess,gpt2,\
        run_name, 'common_nouns', testing=testing, test_cases = test_cases,temperature=temperature)
    print('Took {} seconds to test'.format(time.time() - start))

    file_name = 'results_dic_{}_nouns_{}_objects_{}_temp_{}_run_name.p'.format(
        'test' if testing else 'train',n_objects[0],temperature, run_name
        )
    f = open(file_name, 'wb')
    pickle.dump(results_dic, f)
    f.close()
