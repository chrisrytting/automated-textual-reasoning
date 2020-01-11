import generate_templates
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

def find_accuracy(n_objs, n_containers, sess, gpt2, run_name, test_cases = 10, \
    scenario_type='blocks'):
    if scenario_type == 'blocks':
        truncate = '\n'
    elif scenario_type == 'common_nouns':
        truncate = '<END>'
    
    acc_count = 0.0
    for i in range(test_cases):
        true_scenario = generate_templates.generate_scenario(n_objs, n_containers, \
            scenario_type)
        prefix = re.search('.*Took[^\.]*', true_scenario).group(0)
        predicted_scenario = gpt2.generate_scenario(sess, prefix = prefix, \
            run_name=run_name, truncate =truncate)
        if true_scenario == predicted_scenario:
            acc_count += 1
    return acc_count / test_cases


        
