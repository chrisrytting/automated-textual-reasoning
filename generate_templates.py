import random as random
import t5
import os
import numpy as np


def list_to_nl(list_name, list_in):
    """This function converts a list representing a bin and the blocks it contains to 
    a natural language expression of such.
    
    Arguments:
        list_in {list of ints} -- These represent which blocks are contained in this bin
    
    Returns:
        str -- NL expression of which blocks this bin contains
    """
    # Handling different number of blocks for appropriate gammatical cases
    rep = "The {} contains ".format(list_name)
    if len(list_in) == 0:
        rep += "no objects"
    elif len(list_in) == 1:
        rep += str(list_in[0])
    elif len(list_in) == 2:
        rep += "{} and {}".format(str(list_in[0]), str(list_in[1]))
    else:
        rep += "{}, and {}".format(
            ", ".join([str(i) for i in list_in[:-1]]), str(list_in[-1])
        )
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
    dic = {key: value for (key, value) in zip(np.arange(len(list_names)), list_names)}
    origin_ix = random.choice(np.arange(len(list_names)))
    # If origin_ix list is empty, we want to choose another one to pop from, so we'll keep
    # changing the list while it is empty.
    while len(lists[origin_ix]) == 0:
        origin_ix = (origin_ix + 1) % len(lists)

    # Remove origin_ix from eligible indices so we can choose a different list, not the same one
    # to put the block into
    elig_ix = list(np.arange(len(lists)))
    elig_ix.remove(origin_ix)
    # Draw randomly from the eligible indices to choose a target
    target_ix = np.random.choice(elig_ix)

    # Take random object from origin_ix and put it in target_ix
    obj = random.choice(lists[origin_ix])
    lists[origin_ix].remove(obj)
    lists[target_ix].append(obj)

    # Construct the NL expression of what happened and return both
    rep = "Took {} from the {} and put it into the {}".format(
        obj, dic[origin_ix], dic[target_ix]
    )
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
    # List to hold all NL expressions of modifications
    reps = []

    # Simply perform a random modification n_ops times and store the constructed NL expressions, and return
    for i in range(n_ops):
        lists, rep = generate_and_log_op(lists)
        reps.append(rep)
    return lists, reps


# Randomly generate two lists of integers which represent blocks
def generate_lists(n_objects, n_containers, n_val_nouns=0, test=False):
    """
    Randomly generate two lists of integers which represent blocks

    Arguments:
        n_objects -- number of objects to generate
        n_containers -- number of containers to place objects in
        n_val_nouns {int} -- number of validation nouns to include

    
    Returns:
        tuple of lists -- first bin of blocks and second bin of blocks
    """

    # Generate a number of blocks between 2 and 10 but excluding 5
    lists = [[] for i in range(n_containers)]
    if test:
        filename = "data/little_nounlist_test.txt"
    else:
        filename = "data/little_nounlist_train.txt"

    val_words = []
    if n_val_nouns is not None:
        if n_val_nouns > 0:
            with open("data/little_nounlist_test.txt", "r") as f:
                val_words = f.readlines()
                f.close()
                val_words = random.sample(val_words, n_val_nouns)

    with open(filename, "r") as f:
        # Grab words from file and close it
        all_words = f.readlines()
        f.close()
        # Only keep n_objects of the words and strip whitespace
        kept_words = random.sample(all_words, n_objects)
        if n_val_nouns > 0:
            kept_words = kept_words[:-n_val_nouns]
        kept_words += val_words
        random.shuffle(kept_words)
        words = [word.strip() for word in kept_words]
        # Add proper article a/an
        for word in words:
            if word[0] in list("aeiou"):
                word = "an {}".format(word)
            else:
                word = "a {}".format(word)
            random.choice(lists).append(word)
    return lists


def gen_nl_descriptions(lists, list_names):
    """# Generate a NL description of it
    
    Arguments:
        list_1 {int} -- Bin 1
        list_2 {int} -- Bin 2
    """
    return [list_to_nl(list_names[i], lists[i]) for i in range(len(lists))]


# Perform random operations on that list, coming up with NL descriptions of those operations
def generate_scenario(
    n_objects, n_containers, n_val_nouns=0, testing_conts=False, testing_nouns=False
):
    """Generate random lists, a NL expression describing it, perform an operation on it and describe it in NL, and describe the final state.

    Arguments:
        n_objects {int} -- number of objects to generate
        n_containers {int}-- number of containers to sort objects into

    Returns:
        str -- Description of initial state, action, and final state
    """

    lists = generate_lists(
        n_objects, n_containers, n_val_nouns=n_val_nouns, test=testing_nouns
    )
    list_names = []
    if not testing_conts:
        container_names = "box bin crate tub jar bowl case basket bag".split()
        list_names = random.sample(container_names, n_containers)
    else:
        container_names = "tray sack hole bag room drawer dumpster dish".split()
        list_names = random.sample(container_names, n_containers)
    random.shuffle(list_names)
    is_description = ". ".join(gen_nl_descriptions(lists, list_names))
    fs_lists, action_description = generate_and_log_op(list_names, lists)
    prefix = ". ".join([is_description, action_description])
    target = ". ".join(gen_nl_descriptions(fs_lists, list_names))
    target += ".<END>"
    return prefix, target


def generate_scenarios(n, n_objs, n_conts, save_dir=None):
    prefix_file = open(os.path.join(save_dir, f"prefixes.txt"), "a")
    raw_prefix_file = open(os.path.join(save_dir, f"raw_prefixes.txt"), "a")
    targets_file = open(os.path.join(save_dir, f"targets.txt"), "a")
    for i in range(n):
        raw_prefix, target = generate_scenario(n_objs, n_conts)
        prefix = "initialstate&action: " + raw_prefix

        prefix_file.write(prefix + "\n")
        raw_prefix_file.write(raw_prefix + "\n")
        targets_file.write(target + "\n")


def generate_range_of_scenarios(n, n_objs_range, n_conts_range, experiment):
    # Choose directory whether we're on pccfs or mounted to it in a docker
    # container
    default_exp_dir = "/mnt/pccfs/backed_up/crytting/nlrl/experiments"
    exp_dir = (
        default_exp_dir if os.path.exists(default_exp_dir) else "/nlrl/experiments"
    )
    testing_dir = os.path.join(exp_dir, experiment, "testing")

    # Remove these txt files if they already exist.
    for txtfile in "prefixes.txt raw_prefixes.txt targets.txt".split():
        if os.path.exists(os.path.join(testing_dir, txtfile)):
            os.remove(os.path.join(testing_dir, txtfile))

    for n_objs in n_objs_range:
        for n_conts in n_conts_range:
            generate_scenarios(n, n_objs, n_conts, save_dir=testing_dir)
    desc = (
        f"The prefixes, raw_prefixes, and targets files contained in"
        + " {testing_dir} each contain {n} instances for each value of"
        + " of n_objs in {n_objs_range} and n_conts in {n_conts_range}."
        + " The n_conts increment before the n_objs do e.g. 2 objs, 2"
        + " conts first,"
        + " then 2 objs, 3 conts, etc."
    )
    with open(os.path.join(testing_dir, "desc.txt"), "w") as f:
        f.write(desc)


def predict_from_input_file(
    model,
    experiment,
    input_file,
    output_file,
    checkpoint=-1,
    batch_size=16,
    temperature=0,
):

    model.batch_size = batch_size
    predict_inputs_path = os.path.join(
        "/nlrl/experiments/", experiment, f"testing/{input_file}"
    )
    predict_outputs_path = os.path.join(
        "/nlrl/experiments/", experiment, f"testing/{output_file}"
    )
    model.predict(
        input_file=predict_inputs_path,
        output_file=predict_outputs_path,
        checkpoint_steps=checkpoint,
        # Select the most probable output token at each step.
        temperature=temperature,
    )


def setup_t5_and_predict(
    input_file="prefixes.txt",
    output_file="predictions.txt",
    checkpoint=-1,
    model_parallelism=1,
    batch_parallelism=1,
    gpu_ids=[0],
    experiment="experiment1",
    train_batch_size=16,
    temperature=0.0,
):

    model = t5.models.MtfModel(
        model_dir="/nlrl/models-t5/3B",
        tpu=None,
        mesh_shape=f"model:{model_parallelism},batch:{batch_parallelism}",
        mesh_devices=[f"gpu:{gpu_id}" for gpu_id in gpu_ids],
        batch_size=train_batch_size,
        sequence_length={"inputs": 200, "targets": 200},
        iterations_per_loop=100,
    )

    predict_from_input_file(
        model,
        experiment,
        input_file,
        output_file,
        checkpoint=checkpoint,
        temperature=temperature,
    )


def generate_dataset(n_scenarios, filepath, nolow=2, nohigh=10, nclow=2, nchigh=4):
    """Write n scenarios to a text file with directory filepath
    
    Arguments:
        n_scenarios {int} -- The number of scenarios we want to generate
        filepath {str}    -- The filepath we of the file we want to write
    """
    f = open(filepath, "w")
    for i in range(n_scenarios):
        n_objects = random.randint(nolow, nohigh)
        n_containers = random.randint(nclow, nchigh)
        scenario = generate_scenario(n_objects, n_containers)
        f.write(scenario + "\n")
    f.close()
    print("Successfully generated dataset")


if __name__ == "__main__":
    pass
