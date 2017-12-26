'''
Created on Sep 30, 2017

@author: Adele
'''
import numpy as np

data = np.arange(9).reshape((3,3))
print(data)
data[0,0] = -1
data[1,0] = -3
data[2,0] = -6
for i in range(3): # the last column is Person and is not processed
    print("column number: {}".format(i)) 
    selected_col = data[: , i]
    print("column:\n{}".format(selected_col))
    
    col_min = np.amin(selected_col)
    col_max = np.amax(selected_col)
    print("min: {}, max: {}".format(col_min , col_max))
    
    bins = np.linspace(col_min, col_max, 10)# devide the distance between min and max to 10 parts
    print("bins:\n{}".format(bins))
    
    digitized = np.digitize(selected_col, bins)
    data[: , i] = digitized  
    
    print("digitized column:\n{}".format(data[: , i]))
    
print("final array:\n{}".format(data))
