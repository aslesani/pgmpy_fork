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

#from Validation import calculate_different_metrics


import time
import csv


import pandas as pd
import numpy as np
#from pandas.core.resample import resample

def bic(train,test, scoring_function, name, folder,resultlist,address):
    array=['Person']
    trainstart=time.time()
    #bic=BicScore(train)
    sc = scoring_function(train)
    hc=HillClimbSearch(train, scoring_method=sc)
    best_model=hc.estimate()
    print(best_model.edges())
    #edges=[('c3', 'c2'), ('c3', 'c5'), ('c3', 'c1'), ('c3', 'Person'), ('Person', 'c2'), ('Person', 'c5'), ('Person', 'c1')]
    edges = best_model.edges()
    model=BayesianModel(edges)
    model.fit(train,estimator=BayesianEstimator, prior_type="BDeu")
    trainend=time.time()-trainstart
    
    #for n in model.nodes():
    #    print(model.get_cpds(n))
        

    print("nodes", model.nodes())
    print("column", test.columns)

    flag=0
    if(set(model.nodes())-set(array) ==set(model.nodes())):
        flag=1
    elif(set(model.nodes())-set(array) == set(test.columns)):
        teststart=time.time()
        #print(test)
        result=model.predict(test).values.ravel()
        testend=time.time()-teststart
        pred=list(result)
        #print("y_true: \n" , resultlist , "\ny_predicted:\n" , pred)
    else:
        indicator=list(set(test.columns)-set(model.nodes()))
        print(indicator)
        testchange=test.copy()
        for f in range(len(indicator)):   
            #print(f)
            del testchange[indicator[f]]
        #print(testchange)
        print("come in testchange***********************")
        teststart=time.time()
        result=model.predict(testchange).values.ravel()
        testend=time.time()-teststart
        pred=list(result)
    
    #model_data = XMLBIFWriter(model)
    #model_data.write_xmlbif(address+name+'_bic.bif') 
    if(flag==0):
        fscore,accuracy,precision,recall=calscore(resultlist,pred)
        scores = calculate_different_metrics(y_true = resultlist , y_predicted = pred)
        #draw(model.edges(),name,"bic",folder)
        #WriteData(address+"bicpred\\",name+".xlsx",name,pred)
    else:
        fscore=accuracy=precision=recall=trainend=testend=0
      
    #print("fscore:" , fscore,"accuracy:" ,accuracy,"precision:" ,precision, "recall: ",recall)    
    return (model , scores ,  trainend, testend)

def calculate_different_metrics(y_true , y_predicted):
    f1_score_micro = f1_score(y_true, y_predicted, average='micro') 
    f1_score_macro = f1_score(y_true, y_predicted, average='macro') 
    f1_score_binary = f1_score(y_true, y_predicted, average='binary') 

    precision = precision_score(y_true, y_predicted, average='micro') 
    recall = recall_score(y_true, y_predicted, average='micro')
    accuracy = accuracy_score(y_true, y_predicted)  
    
    scores = {'f1_score_micro': f1_score_micro, 
              'f1_score_macro': f1_score_macro,
              'f1_score_binary': f1_score_binary,
              'precision' : precision,
              'recall' : recall,
              'accuracy' : accuracy
              }
    
    return scores

   
def calscore(result,predict):
    fscore=f1_score(result,predict)
    accuracy=accuracy_score(result,predict)
    precision=precision_score(result,predict)
    recall=recall_score(result,predict)
    
    
    return fscore,accuracy,precision,recall


def read_Abdoolahi_data():
    dest_file = r"C:\f5_0_10.csv"
    
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        data = [data for data in data_iter]
    
    
    numpy_result = np.asarray(data, dtype = np.int)
   
    _ , cols = numpy_result.shape
    column_names = []

    for names in range(1, cols):
        column_names.append('c' + str(names))
    column_names.append("Person")  
       
    panda_result = pd.DataFrame(data=numpy_result , columns= column_names , dtype = np.int) 
    
    print(panda_result.columns)
    #print(panda_result)
    
    return panda_result
    
    #return numpy_result


if __name__ == "__main__":
    
    data = read_Abdoolahi_data()
    #data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), 
    #                    columns=['A', 'B', 'C', 'D', 'res'])
    
    #print(data)
    train = data[0:700] 
    test = data[700:]
    resultlist = test['Person'].values
    test = test.drop('Person', axis=1, inplace=False)
    #scores , learning_time = 
    bic(train = train, test = test, scoring_function=BicScore , name = "model", 
        folder = "Abdollahi", resultlist = resultlist, address = "C:\\")
    
    #print(scores , learning_time)
    