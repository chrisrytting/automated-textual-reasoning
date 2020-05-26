import numpy as np

def generate_word_problems(
        n_scenarios = 500,
        lines_file = 'lines.txt',
        names_file = 'names.txt',
        objects_file = 'objects.txt',
        data_dir = 'data/wordprobs'):
    #TODO   Get the "answers" into prefix and target files.
    #Training set, scenarios with even objects up to 50
    #Interpolation validation set, scenarios with odd objects up to 50
    #Extrapolation validation set, scenarios with
     
     
    training_choices = np.arange(1,51,2)
    interp_choices = np.arange(0,52,2)
    extrap_choices = np.arange(51,101)
    print(training_choices, interp_choices, extrap_choices)

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
            lfile.write(word_problem + " " + exact_equation
