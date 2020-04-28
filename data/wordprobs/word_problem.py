import numpy as np

base_dir = '/mnt/pccfs/backed_up/crytting/nlrl'
if not os.exists(base_dir):
    base_dir = '/nlrl'

data_dir = os.path.join(base_dir, 'data/wordprobs/')
names_file = 'names.txt'
objects_file = 'objects.txt'

with open(os.path.join(data_dir,names_file), "r") as f:
    names = f.read().split("\n")[:-1]

with open(os.path.join(data_dir,objects_file), "r") as f:
    objects = f.read().split("\n")[:-1]


n_trials = 500

for i in range(n_trials):
    problem_types = ["+", "-"]

    person1, person2 = np.random.choice(names,2)

    object = np.random.choice(objects)

    n = np.random.randint(1, 100)
    c = np.random.randint(0, n)

    problem_type = np.random.choice(problem_types)

    if n > 1:
        object += "s"
    if problem_type == "+":
        word_problem = f"{person1} has {n} {object}. Then {person2} gives {person1} {c} more. How many {object} does {person1} have?"

        exact_equation = f"{n} + {c} = {n + c}"
    elif problem_type == "-":
        word_problem = f"{person1} has {n} {object}. Then {person2} takes {c} from {person1}. How many {object} does {person1} have?"

        exact_equation = f"{n} - {c} = {n - c}"

    print(word_problem, exact_equation)
