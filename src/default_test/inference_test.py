'''
Created on Sep 21, 2017

@author: Adele
'''

from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation

from pgmpy.models import BayesianModel

import pgmpy.factors.discrete.DiscreteFactor

import numpy as np
import pandas as pd
import time
from _tracemalloc import start
from pgmpy.factors.discrete.DiscreteFactor import DiscreteFactor

values = pd.DataFrame(np.random.randint(low=0, high=2, size=(100000, 5)),
                       columns=['A', 'B', 'C', 'D', 'E'])

print(values)
model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
model.fit(values)
ve = VariableElimination(model)
bp = BeliefPropagation(model)

start = time.time()
ve_result = ve.query(['B'] , evidence={'A':0})#, 'B'])
end = time.time()
print(ve_result['B'])
print("time: {}".format(end-start))


start = time.time()
bp_result = bp.query(['B'] , evidence={'A':0})

'''
if 'B' in bp_result:
    if 'B_0' in bp_result['B']:
        print("yes")
    else:
        print("no")
''' 

print(bp_result['B'].variables)
print(bp_result['B'].values)



'''for key, value in bp_result.items():
    print value.keys()[0]
   '''     
      

'''
end = time.time()
print(type(bp_result['B']))
print(type(bp_result['B'].values()))

print(bp_result['B'])
print("time: {}".format(end-start))
'''
