import random as random

def list_to_nl(list_in):
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

# Randomly generate two lists of integers which represent blocks 
n = random.randint(0,20)
list_1 = []
list_2 = []
for i in range(n):
    random.choice((list_1, list_2)).append(i)
#print(list_1,list_2)

# Generate a NL description of it
# description_1 = 

# Perform random operations on that list, coming up with NL descriptions of those operations 

# Generate a NL description of the manipulated list

if __name__ == "__main__":
    block_list = [2,4,5]
    nl = list_to_nl(block_list)
    print(nl)
    