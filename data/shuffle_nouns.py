import random
lines = open('nounlist.txt').readlines()
random.shuffle(lines)
open('nounlist.txt', 'w').writelines(lines)