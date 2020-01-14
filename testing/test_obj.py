import generate_templates as gt
import re

def find_accuracies(n_objs_to_test, n_containers_to_test, sess, gpt2, run_name):
    """Find accuracies on a test set for several n_objs and n_containers
    
    Arguments:
        n_objs_to_test {list{ints}} -- list of different n_objs wanting to test for
        n_containers_to_test {list{ints}} -- list of different n_containers wanting to test for
        
    
    Returns:
        np.array -- accuracies for each of the n_objs on axis 0 and n_containers on axis 1
    """
    accuracies = np.ones(len(n_objs_to_test), len(n_containers_to_test)) * -1
    for i, n_objs in enumerate(n_objs_to_test):
        for j, n_containers in enumerate(n_containers_to_test):
            acc = find_accuracy(n_objs,n_containers, gpt2, run_name)
            accuracies[i,j] = acc
    return accuracies

def find_accuracy(n_objs, n_containers, sess, gpt2, run_name, scenario_type='blocks', \
    test_cases = 10, testing=False):
    truncate = '<END>'
    acc_count = 0.0
    result_dic = {}
    for i in range(test_cases):
        true_scenario = gt.generate_scenario(n_objs, n_containers, \
            scenario_type)
        prefix = re.search('.*Took[^\.]*', true_scenario).group(0)
        predicted_scenario = gpt2.generate(sess, prefix = prefix, \
            run_name=run_name, truncate =truncate,return_as_list=True)[0] + truncate
        match = true_scenario == predicted_scenario

        #Log results in a dic
        result_dic['true_scenario_{}'.format(i)] = true_scenario
        result_dic['prefix_{}'.format(i)] = prefix
        result_dic['predicted_scenario_{}'.format(i)] = predicted_scenario
        result_dic['match_{}'.format(i)] = match
        if match:
            acc_count += 1
    score = acc_count / test_cases
    result_dic['score' = score]
    return score, result_dic


        

if __name__=="__main__":
    import gpt_2_simple as gpt2

    sess = gpt2.start_tf_sess()
    run_name = 'common_nouns'
    gpt2.load_gpt2(sess,run_name=run_name)
    find_accuracy(5,4,sess,gpt2,)