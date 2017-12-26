'''
Created on July 12, 2017

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
#feature_names = dataPreparation.get_work_lists()
#feature_names.append("Person")
#print(feature_names)
#mydata = np.random.randint(low=0, high=2,size=(100, 6))
mydata = np.genfromtxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Aras\House A\CSV_Summery\Sequential\Day\occur\Whole_data.csv', delimiter=",")
#pd.read_csv(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\data.csv')
#print(mydata)
feature_names = [str(i) for i in range (1,41)]
feature_names.append("Person")
feature_names.append("activity")
print(feature_names)
data = pd.DataFrame(mydata, columns= feature_names)#['X', 'Y'])
print(data)

list_of_scoring_methods = [#BicScore(data),
                           #BdeuScore(data),
                           K2Score(data)]

for scoreMethod in list_of_scoring_methods:
    start_time = time.time()
    hc = HillClimbSearch(data, scoreMethod)
    best_model = hc.estimate()
    print(hc.scoring_method)
    print(best_model.edges())
    end_time = time.time()
    print("execution time in seconds:")
    print(end_time-start_time)



#casas7_model = BayesianModel()
#casas7_model.fit(data, estimator=BayesianEstimator)#MaximumLikelihoodEstimator)
#print(casas7_model.get_cpds())
#casas7_model.get_n