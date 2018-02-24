'''
Created on Jan 26, 2018

@author: Adele
'''
import numpy as np
from pgmpy.models import BayesianModel
import pandas as pd
from DimensionReductionandBNStructureLearning import create_BN_model
from Validation import pgm_test
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore



data = np.random.randint(low=0, high=2, size=(1000, 3))
#print(data)
data = pd.DataFrame(data, columns= ['cost', 'location', 'no_of_people'])#['cost', 'quality', 'location', 'no_of_people'])
#print(data.loc[:,'no_of_people'])
train = data[:750]
y_true = data[750:]['no_of_people'].values
#print("y_ture: " , y_true)
test1 = data[750:]
test = data[750:].drop('no_of_people', axis=1)
#estimator , _ = create_BN_model(train)
#pgm_test(estimator, test_set = test , target_column_name = 'no_of_people')


restaurant_model = BayesianModel(
[('location', 'cost'),
#('quality', 'cost'),
('location', 'no_of_people'),
('cost', 'no_of_people')])

for est in [BayesianEstimator]:#MaximumLikelihoodEstimator
    restaurant_model.fit(train ,estimator= est)
    #restaurant_model.get_cpds()
    a1 = restaurant_model.predict(test).values.ravel()
    a2 = pgm_test(restaurant_model, test_set = test1, target_column_name = 'no_of_people')
    print(est,'\n', a1 ,'\n' , a2)


