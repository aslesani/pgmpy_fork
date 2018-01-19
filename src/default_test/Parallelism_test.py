'''
Created on Jan 16, 2018

@author: Adele
'''

from math import sqrt
from joblib import Parallel, delayed

def my_func(i,j):
    print("i , j: " , i , j)
    return (i+1 , j+1)


if __name__ == "__main__":
    
    a = Parallel(n_jobs=20 , backend="threading")(delayed(my_func)(i , j) for i,j in zip(range(10), range(100,110)))
    print(type(a))
    