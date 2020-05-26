import os

def takeoffix(line):
    return " ".join(line.split()[1:])

def tomtoatr(tomdir, tomfile, atrdir, atrfile):
    lines = open(os.path.join(tomdir,tomfile),'r').readlines()
    lines = [line.strip() for line in lines]
    substories = []
    counter = 0
    story = []
    lastline = None
    for line in lines:
        counter += 1

        if '?' in line:
            lastline = takeoffix(line)
            split = lastline.split('?')
            story.append(f'{split[0]}?\t{split[1]}')
            substories.append(story.copy())
            story.pop()
        elif line.split()[0] == str(counter):
            story.append(takeoffix(line) + '. ')
        else:
            counter = 1
            story = [takeoffix(line)]
    f = open(os.path.join(atrdir, atrfile), 'w')
    for substory in substories:
        f.write(' '.join(substory) + '\n')


        



if __name__=="__main__":
    #Take files in raw data and get them into right format for T5
    tomtoatr('data/tomprobs/rawdata/tom/world_large_nex_1000_0/','qa21_task_AB_train.txt', 'data/tomprobs/','atr-train-0.txt')
    tomtoatr('data/tomprobs/rawdata/tom/world_large_nex_1000_10/','qa21_task_AB_train.txt', 'data/tomprobs/','atr-train-10.txt')
    tomtoatr('data/tomprobs/rawdata/tom_easy/world_large_nex_1000_0/','qa21_task_AB_train.txt', 'data/tomprobs/','atr-train-easy-0.txt')
    tomtoatr('data/tomprobs/rawdata/tom_easy/world_large_nex_1000_10/','qa21_task_AB_train.txt', 'data/tomprobs/','atr-train-easy-10.txt')
