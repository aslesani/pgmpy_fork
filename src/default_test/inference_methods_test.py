'''
Created on Jan 23, 2018

@author: Adele
'''
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianModel
import numpy as np
import pandas as pd
import time

values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                       columns=['A', 'B', 'C', 'D', 'E'])
model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
model.fit(values)


for method in [VariableElimination , BeliefPropagation]:
    start_time = time.time()
    inference = method(model)
    phi_query = inference.map_query(variables=['A', 'B'] , evidence={'C': 0 , 'D': 0})
    phi_query2 = inference.query(variables=['A', 'B'] , evidence={'C': 0 , 'D': 0})

    end_time = time.time()
    
    print(phi_query , phi_query2 ,end_time - start_time)
