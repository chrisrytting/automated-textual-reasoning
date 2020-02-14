import random as random
import numpy as np

def list_to_nl(list_name, list_in):
    """This function converts a list representing a bin and the blocks it contains to 
    a natural language expression of such.
    
    Arguments:
        list_in {list of ints} -- These represent which blocks are contained in this bin
    
    Returns:
        str -- NL expression of which blocks this bin contains
    """
    #Handling different number of blocks for appropriate gammatical cases
    rep = "The {} contains ".format(list_name)
    if len(list_in) == 0:
        rep += "no objects"
    elif len(list_in) == 1:
        rep += str(list_in[0])
    elif len(list_in) == 2:
        rep += "{} and {}".format(str(list_in[0]), str(list_in[1]))
    else:
        rep += "{}, and {}".format(", ".join([str(i) for i in list_in[:-1]]),\
            str(list_in[-1]))
    return rep

def generate_and_log_op(list_names, lists):
    """This function generates a random operation on a list of lists and returns both the
    modified lists and the NL expression of the operation.
    
    Arguments:
        lists {list of lists of ints} -- These are lists to be operated on, which operations
        are then logged
    
    Returns:
        list of lists of ints, str -- The modified lists, The NL expression of which 
        modifications were made.
    """
    # Generate a random list to remove a block from and a random list to place a block into, 
    # which can be any of the lists besides the one we started with.
    dic = {key:value for (key,value) in zip(np.arange(len(list_names)),list_names)}
    origin_ix = random.choice(np.arange(len(list_names)))
    #If origin_ix list is empty, we want to choose another one to pop from, so we'll keep
    #changing the list while it is empty.
    while len(lists[origin_ix]) == 0:
        origin_ix = (origin_ix + 1) % len(lists)

    #Remove origin_ix from eligible indices so we can choose a different list, not the same one
    #to put the block into
    elig_ix = list(np.arange(len(lists)))
    elig_ix.remove(origin_ix)
    #Draw randomly from the eligible indices to choose a target
    target_ix = np.random.choice(elig_ix)

    #Take random object from origin_ix and put it in target_ix
    obj = random.choice(lists[origin_ix])
    lists[origin_ix].remove(obj)
    lists[target_ix].append(obj)


    #Construct the NL expression of what happened and return both
    rep = 'Took {} from the {} and put it into the {}'.format(obj, dic[origin_ix], dic[target_ix])
    return lists, rep

def generate_and_log_ops(n_ops, lists):
    """This function takes a list of lists and performs generate_and_log_op on it n_ops times.
    
    Arguments:
        n_ops {int} -- number of times to perform generate_and_log_op on lists
        lists {list of lists of ints} -- bins containing blocks, to be operated on
    
    Returns:
        list of list of ints, list of str -- the modified lists, a list of NL expressions of 
        modifications of the lists explaining what happened in the modifications.
    """
    #List to hold all NL expressions of modifications
    reps = [] 

    #Simply perform a random modification n_ops times and store the constructed NL expressions, and return
    for i in range(n_ops):
        lists, rep = generate_and_log_op(lists)
        reps.append(rep)
    return lists, reps
        


# Randomly generate two lists of integers which represent blocks 
def generate_lists(n_objects, n_containers, test=False):
    """Randomly generate two lists of integers which represent blocks

    Arguments:
        n_objects -- number of objects to generate
        n_containers -- number of containers to place objects in

    
    Returns:
        tuple of lists -- first bin of blocks and second bin of blocks
    """

    #Generate a number of blocks between 2 and 10 but excluding 5
    lists = [[] for i in range(n_containers)]
    if test:
        filename = 'data/little_nounlist_test.txt'
    else:
        filename = 'data/little_nounlist_train.txt'

    with open(filename, 'r') as f:
        #Grab words from file and close it
        all_words = f.readlines()
        f.close()
        #Only keep n_objects of the words and strip whitespace
        kept_words = random.sample(all_words, n_objects)
        words = [word.strip() for word in kept_words]
        #Add proper article a/an
        for word in words:
            if word[0] in list('aeiou'):
                word = 'an {}'.format(word)
            else:
                word = 'a {}'.format(word)
            random.choice(lists).append(word)
    return lists

def gen_nl_descriptions(lists,list_names):
    """# Generate a NL description of it
    
    Arguments:
        list_1 {int} -- Bin 1
        list_2 {int} -- Bin 2
    """
    return [list_to_nl(list_names[i], lists[i]) for i in range(len(lists))]

# Perform random operations on that list, coming up with NL descriptions of those operations 
def generate_scenario(n_objects,n_containers,testing_conts=False,
    testing_nouns=False):
    """Generate random lists, a NL expression describing it, perform an operation on it and describe it in NL, and describe the final state.

    Arguments:
        n_objects {int} -- number of objects to generate
        n_containers {int}-- number of containers to sort objects into

    Returns:
        str -- Description of initial state, action, and final state
    """
    
    lists = generate_lists(n_objects,n_containers, test=testing_nouns)
    list_names = []
    if not testing_conts:
        container_names = 'box bin crate tub jar bowl case basket bag'.split()
        list_names = random.sample(container_names,n_containers)
    else:
        container_names = 'tray sack hole bag room drawer dumpster dish'.split()
        list_names = random.sample(container_names,n_containers)
    random.shuffle(list_names)
    is_description = '. '.join(gen_nl_descriptions(lists, list_names))
    fs_lists, action_description = generate_and_log_op(list_names, lists)
    fs_description = '. '.join(gen_nl_descriptions(fs_lists, list_names))
    description = '. '.join([is_description, action_description, fs_description])
    description += '.<END>'
    return description

def generate_dataset(n_scenarios):
    """Write n scenarios to a text file
    
    Arguments:
        n {int} -- number of lines to write to a text file
        n_scenarios {int} -- The number of scenarios we want to generate
    """
    f = open('data/little_objworld.txt', 'w')
    for i in range(n_scenarios):
        n_objects = random.randint(2,10)
        n_containers = random.randint(2,4)
        scenario = generate_scenario(n_objects,n_containers)
        f.write(scenario + '\n')
    f.close()
    print('Successfully generated dataset')

if __name__=="__main__":
    import time
    start = time.time()
    generate_dataset(int(5e5))
    print("Took {} seconds".format(time.time() - start))
    

    #lists = generate_lists(10,2,'common_nouns')
    #print(lists)
    #print(list_to_nl('Bin', lists[0]))
    #print(list_to_nl('Box',lists[1]))
    
