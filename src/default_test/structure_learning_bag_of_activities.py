'''
Created on May 31, 2017

@author: Adele
'''
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
import csv
import time
import dataPreparation# import get_work_lists

#print(dataPreparation.get_work_lists())
feature_names = dataPreparation.get_work_lists()
feature_names.append("Person")
print(feature_names)
#mydata = np.random.randint(low=0, high=2,size=(100, 6))
mydata = np.genfromtxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities+time_ordered_withoutdatetime.csv', delimiter=",")
#pd.read_csv(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\data.csv')
#print(mydata)
data = pd.DataFrame(mydata, columns= feature_names)#['X', 'Y'])
print(data)

list_of_scoring_methods = [BicScore(data),
                           #BdeuScore(data),
                           #K2Score(data)
                           ]

for scoreMethod in list_of_scoring_methods:
    start_time = time.time()
    hc = HillClimbSearch(data, scoreMethod)
    best_model = hc.estimate()
    print(hc.scoring_method)
    print(best_model.edges())
    end_time = time.time()
    print("execution time in seconds:")
    print(end_time-start_time)


estimator = BayesianEstimator(best_model, data)
print(estimator.get_parameters(prior_type='K2'))#, equivalent_sample_size=5)
    

#casas7_model = BayesianModel()
#casas7_model.fit(data, estimator=BayesianEstimator)#MaximumLikelihoodEstimator)
#print(casas7_model.get_cpds())
#casas7_model.get_n