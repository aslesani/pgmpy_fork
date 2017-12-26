'''
Created on September 02, 2017

@author: Adele
'''

import numpy as np
import pandas as pd
from DimensionReductionandBNStructureLearning import create_BN_model, read_data_from_PCA_digitized_file, discretization_equal_width_for_any_data

import matplotlib.pyplot as plt

from sklearn import datasets


from pgmpy.inference import BeliefPropagation
from pgmpy.inference import VariableElimination

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from default_test.DimensionReductionandBNStructureLearning import create_BN_model_using_BayesianEstimator

def create_BN_model(data): 
    '''
    create a Bayesian Model (structure learning and Parameter learning)
    
    Parameters:
    ==========
    data: the data which we want to create a bn for
    
    Returns:
    =======
    model:
    learning_time:
    
    '''   
    data = pd.DataFrame(data)
    print("structure learning")
    hc = HillClimbSearch(data, scoring_method= K2Score(data))#BicScore(data))#K2Score(data))BdeuScore(data)
    best_model = hc.estimate()

    casas7_model = BayesianModel(best_model.edges())
    casas7_model.fit(data, estimator=BayesianEstimator, prior_type="K2")
    return casas7_model



def kfoldcrossvalidationForBNModel_UsingPanda(k , data, target_column_name, scoring='f1_micro' ):  
    '''
    Parameters:
    -------
    k: the value of k in k-fold
    data: a pandaFrame object the estimator_model would be tested on 
          (we imagine the last column is the target)
          
    scoring: the scoring method , default value:'f1_micro'
    
    Returns
    -------
    scores: a vector of scores, the length of the vector is k.
    
    ''' 
    data_length = np.shape(data)
    part_length = int(data_length / k)
    
    partition = np.zeros(k , dtype= pd.DataFrame)
    index = 0
    for i in range(0, k): 
        end = (i+1) * part_length
        partition[i] = data[index:end]
        index = end
    
    scores = np.zeros(k)
    
    for i in range(0,k):
        test_set = partition[i]
        train_set =  np.delete(partition, i, 0)
        pd_train_set = train_set[0]
        for j in range(1, k-1):
            pd_train_set = pd_train_set.append(train_set[j]) 
        
        estimator = create_BN_model(pd_train_set)
        
        scores[i] = pgm_test(estimator, test_set, target_column_name)
        
    return scores

    
def pgm_test(estimator, test_set, target_column_name):
    '''
    test the pgm model. the model is trained by a train_set. 
    
    Parameters:
    =========== 
    estimator: the pgm model that should be tested 
    
    test_set: clear ;) the type of it is panda dataframe
    
    target_column_name: the name of the column that should be predicted
    
    Return:
    =======
    score
    '''
    y_true = test_set[[target_column_name]]#, y_predicted = np.zeros(1, number_of_tests)
    
    test_set = test_set.drop(target_column_name, axis=1, inplace=False)

    y_predicted = y_true
    #print(type(y_true))
    rows , _ = test_set.shape
    #print(rows)
    for i in range (0 ,rows):
        print("i:{}".format(i))
        #if i < 10:
        #    print("test_set.iloc[[i]]:{}".format(test_set.iloc[[i]]))
        #if i >37:
        #   print("start trace from here!")
        #print(type(estimator))
        print("test data:\n {}".format(test_set.iloc[[i]]))
        y_predicted.iloc[[i]] = estimator.predict(test_set.iloc[[i]])
    
    #print("y_true:\n{}\ny_predicted:\n{}".format( y_true, y_predicted))
    score = f1_score(y_true, y_predicted, average='micro') 
    return score

    
def iris_dicretization():
    
    iris = datasets.load_iris()
    data = iris.data#[:, :2]  # we only take the first two features. We could
                        # avoid this ugly slicing by using a two-dim dataset
    target = iris.target
    
    target.shape = (150,1)
    
    digitized_data = discretization_equal_width_for_any_data(np.concatenate((data, target), axis=1))
    
    for i in range(0,5):
        #print(set(digitized_data[:, i]))
        digits_set = list(set(digitized_data[:, i]))
        if len(digits_set) < 10:
            for row in range(0,150):
                digitized_data[row , i] = digits_set.index(digitized_data[row , i])
        elif 0 not in digits_set:
            digitized_data[:, i] -=1
            
        print(set(digitized_data[:, i]))   
    
    pd_data = pd.DataFrame( digitized_data , columns = ['c1' , 'c2' , 'c3' , 'c4' , 'target'])

    return pd_data
       
    
if __name__ == '__main__':
    
    data = iris_dicretization()
    print(data)
    print(kfoldcrossvalidationForBNModel_UsingPanda(10, data, target_column_name = "target", scoring = "f1_micro"))#data.iloc[0:100 ,:]
    