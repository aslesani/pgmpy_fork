'''
Created on Sep 30, 2017

@author: Adele
'''
import numpy as np

x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
bins = np.array([0, 5, 10, 15, 20])
right_digitized = np.digitize(x,bins,right=True)
print("right_digitized:{}".format(right_digitized))

left_digitized = np.digitize(x,bins,right=False)
print("left_digitized:{}".format(left_digitized))

