import random as random
import numpy as np

def list_to_nl(list_in):
    """This function converts a list representing a bin and the blocks it contains to 
    a natural language expression of such.
    
    Arguments:
        list_in {list of ints} -- These represent which blocks are contained in this bin
    
    Returns:
        str -- NL expression of which blocks this bin contains
    """
    #Handling different number of blocks for appropriate gammatical cases
    rep = "Bin contains "
    if len(list_in) == 0:
        rep += "no blocks"
    elif len(list_in) == 1:
        rep += "block " + str(list_in[0])
    elif len(list_in) == 2:
        rep += "blocks {} and {}".format(str(list_in[0]), str(list_in[1]))
    else:
        rep += "blocks {}, and {}".format(", ".join([str(i) for i in list_in[:-1]]),\
            str(list_in[-1]))
    return rep

def generate_and_log_op(lists):
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

    origin_ix = random.randint(0,len(lists) - 1)
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

    #Take block from origin_ix and put it in target_ix
    block = lists[origin_ix].pop()
    lists[target_ix].append(block)

    #Construct the NL expression of what happened and return both
    rep = 'Took block {} from bin {} and put it into bin {}'.format(block, origin_ix, target_ix)
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
n = random.randint(0,20)
list_1 = []
list_2 = []
for i in range(n):
    random.choice((list_1, list_2)).append(i)

# Generate a NL description of it
description_1 = list_to_nl(list_1)
description_2 = list_to_nl(list_2)
print(description_1, description_2)

# Perform random operations on that list, coming up with NL descriptions of those operations 
n_operations = random.randint(1,10)
lists = [list_1,list_2]
generate_and_log_op(lists)


# Generate a NL description of the manipulated list
