import os

def tomtoatr(tomdir, tomfile, atrdir, atrfile):
    lines = open(os.path.join(tomdir,tomfile),'r').readlines()
    lines = [line.strip() for line in lines]
    newlines = []
    counter = 0
    newline = ""
    lastline = None
    for line in lines:
        if line.split()[0] == str(counter + 1):
            newline += "(" + line + ")"
            counter += 1
        else:
            counter = 0
            newlines.append(newline)
            newline = line
        lastline = line
    f = open(os.path.join(atrdir, atrfile), 'w')
    for newline in newlines:
        f.write(newline + '\n')


        



if __name__=="__main__":
    #Take files in raw data and get them into right format for T5
    tomtoatr('data/tomprobs/rawdata/tom/world_large_nex_1000_0/','qa21_task_AB_train.txt', 'data/tomprobs/','atr-train.txt')
