import numpy as np

def generate_word_problems(
        n_scenarios = 500,
        experiment_type='interp_train',
        targets_file = 'targets.txt',
        prefixes_file = 'prefixes.txt',
        lines_file = 'lines.txt',
        names_file = 'names.txt',
        objects_file = 'objects.txt',
        data_dir = 'data/wordprobs',
        base_dir = '/mnt/pccfs/backed_up/crytting/nlrl'):
    #TODO   Get the "answers" into prefix and target files.
    #       What is the training set?
    #       What is the validation set?

    # there are 4 experiment_types interp_train, interp_val,
    # extrap_train, extrap_val. These are parameterized by
    # 

    if not os.exists(base_dir):
        base_dir = '/nlrl'
    
    data_dir = os.path.join(base_dir, data_dir)
    
    with open(os.path.join(data_dir,names_file), "r") as f:
        names = f.read().split("\n")[:-1]
    
    with open(os.path.join(data_dir,objects_file), "r") as f:
        objects = f.read().split("\n")[:-1]
    
    
    problem_types = ["+", "-"]

    with open(os.path.join(data_dir, targets_file),'w'),\
         open(os.path.join(data_dir, lines_file),'w'),\
         open(os.path.join(data_dir, prefixes_file),'w') as tfile, lfile, pfile:
        for i in range(n_scenarios):
        
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


        
            pfile.write(word_problem)
            tfile.write(exact_equation)
            lfile.write(word_problem + " " + exact_equation)

