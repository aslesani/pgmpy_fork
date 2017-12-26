'''
Created on May 14, 2017

@author: Adele
'''
from pgmpy.estimators import HillClimbSearch
import pandas as pd
import numpy as np
from pgmpy.estimators import BdeuScore, K2Score, BicScore


# create some data with dependencies
data = pd.DataFrame(np.random.randint(0, 3, size=(2500, 8)), columns=list('ABCDEFGH'))
data['A'] += data['B'] + data['C']
data['H'] = data['G'] - data['A']
#print(data)

hc = HillClimbSearch(data, scoring_method=BicScore(data))

best_model = hc.estimate()
print(hc.scoring_method)
print(best_model.edges())


hc = HillClimbSearch(data, scoring_method=BdeuScore(data))
best_model = hc.estimate()
print(hc.scoring_method)
print(best_model.edges())

hc = HillClimbSearch(data, scoring_method=K2Score(data))
best_model = hc.estimate()
print(hc.scoring_method)
print(best_model.edges())

