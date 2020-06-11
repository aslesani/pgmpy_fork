import random
from operator import add

def generate(n):
    diff = []
    for i in range(19):
        diff.append(-1 * random.randint(0, 7)/10)
    
    a = list(map(add,[n]*19, diff))
    return [round(i,2) for i in a]