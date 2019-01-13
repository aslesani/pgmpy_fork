# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:20:20 2018

@author: Adele
"""

import numpy as np

element = 2*np.arange(4).reshape((2, 2))
print(element)


test_elements = [1, 2, 4, 8]
mask = np.isin(element, test_elements)
print(mask)

print(element)

#print(element[mask])