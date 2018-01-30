'''
Created on Jan 27, 2018

@author: Adele
'''
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *

from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianModel
from pgmpy.readwrite import XMLBIFWriter

from Validation import read_Abdoolahi_data

import time

import pandas as pd
import numpy as np

def bic(train,test,name,folder,resultlist,address):
    array=['res']
    trainstart=time.time()
    bic=BicScore(train)
#    bdeu=BdeuScore(train,equivalent_sample_size=5)
    hc=HillClimbSearch(train, scoring_method=bic)
    best_model=hc.estimate()
    print(best_model.edges())
    edges=best_model.edges()
    model=BayesianModel(edges)
    model.fit(train,estimator=BayesianEstimator, prior_type="BDeu")
    trainend=time.time()-trainstart
    
    for n in model.nodes():
        print(model.get_cpds(n))
        
    print("nodes", model.nodes())
    print("column", test.columns)

    flag=0
    if(set(model.nodes())-set(array) ==set(model.nodes())):
        flag=1
    elif(set(model.nodes())-set(array) == set(test.columns)):
        teststart=time()
        result=model.predict(test).values.ravel()
        testend=time()-teststart
        pred=list(result)
        print("y_true: \n" , resultlist , "\ny_predicted:\n" , pred)
    else:
        indicator=list(set(test.columns)-set(model.nodes()))
        print("indicator:\n" , indicator)
        testchange=test.copy()
        print(testchange)

        for f in range(len(indicator)):   
            print(f)
            del testchange[indicator[f]]
        print("after: \n" , testchange)
        teststart=time.time()
        result=model.predict(testchange).values.ravel()
        testend=time.time()-teststart
        pred=list(result)
        print("y_true: \n" , resultlist , "\ny_predicted:\n" , pred)

    
    model_data = XMLBIFWriter(model)
    model_data.write_xmlbif(address+name+'_bic.bif') 
    if(flag==0):
        fscore,accuracy,precision,recall=calscore(resultlist,pred)
        #draw(model.edges(),name,"bic",folder)
        #WriteData(address+"bicpred\\",name+".xlsx",name,pred)
    else:
        fscore=accuracy=precision=recall=trainend=testend=0
      
    print(fscore,accuracy,precision,recall)    
        
def calscore(result,predict):
    fscore=f1_score(result,predict)
    accuracy=accuracy_score(result,predict)
    precision=precision_score(result,predict)
    recall=recall_score(result,predict)
    return fscore,accuracy,precision,recall

if __name__ == "__main__":
    
    data = read_Abdoolahi_data()
    #data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'res'])
    
    train = data[0:700] 
    test = data[700:]
    resultlist = test['res'].values
    test = test.drop('res', axis=1, inplace=False)
    bic(train = train, test = test, name = "model", 
        folder = "Abdollahi", resultlist = resultlist, address = "E:\\")