'''
Created on Jan 27, 2018

@author: Adele
'''

from sklearn.metrics import f1_score, precision_score , recall_score, accuracy_score

from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianModel
from pgmpy.readwrite import XMLBIFWriter
from DimensionReductionandBNStructureLearning import shift_each_column_separately
from data_utils import check_data


#from Validation import calculate_different_metrics


import time
import csv


import pandas as pd
import numpy as np
#from pandas.core.resample import resample

def bic(train,test, scoring_function,resultlist):
    #print(set(train['Person'].values))
    #print(set(train['c0'].values))
    #print(set(train['c1'].values))

    #print(len(test))
    #print('################')
    array=['Person']
    trainstart=time.time()
    #bic=BicScore(train)
    sc = scoring_function(train)
    hc=HillClimbSearch(train, scoring_method=sc)
    best_model=hc.estimate()
    #print("best_model.edges:" , best_model.edges())

    #edges=[('c3', 'c2'), ('c3', 'c5'), ('c3', 'c1'), ('c3', 'Person'), ('Person', 'c2'), ('Person', 'c5'), ('Person', 'c1')]
    edges = best_model.edges()
    model=BayesianModel(edges)
    model.fit(train,estimator=BayesianEstimator, prior_type="BDeu")
    trainend=time.time()-trainstart
    

    #for n in model.nodes():
    #    print(model.get_cpds(n))
        

    #print("nodes:", model.nodes())
    #print("test column:", test.columns)

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
        #print("indicator:\n" , indicator)
        #print("come in testchange***********************")
        #print("before cahnge:" , len(test))
        testchange=test.copy()
        #print(testchange)

        for f in range(len(indicator)):   
            #print(f)
            del testchange[indicator[f]]
        #print(testchange)
        #print("after cahnge:" , len(testchange))

        teststart=time.time()
        result=model.predict(testchange).values.ravel()
        testend=time.time()-teststart
        pred=list(result)
        #print("y_true: \n" , resultlist , "\ny_predicted:\n" , pred)

    
    #model_data = XMLBIFWriter(model)
    #model_data.write_xmlbif(address+name+'_bic.bif') 
    if flag == 1:
        print('##############flag:' , flag)
    if(flag==0):
        fscore,accuracy,precision,recall=calscore(resultlist,pred)
        scores = calculate_different_metrics(y_true = resultlist , y_predicted = pred)
        #draw(model.edges(),name,"bic",folder)
        #WriteData(address+"bicpred\\",name+".xlsx",name,pred)
    else:
        fscore=accuracy=precision=recall=trainend=testend=0
        scores = {'f1_score_micro': 0, 
              'f1_score_macro': 0,
              'f1_score_binary': 0,
              'precision' : 0,
              'recall' : 0,
              'accuracy' : 0
              }
    
    #print("set(pred)", set(pred))
    #print("set(resultlist):", set(resultlist))
    #print("fscore:" , fscore,"accuracy:" ,accuracy,"precision:" ,precision, "recall: ",recall)    
    #print("scores:", scores)
    return (model , scores ,  trainend, testend, pred)

def calculate_different_metrics(y_true , y_predicted):
    f1_score_micro = f1_score(y_true, y_predicted, average='micro') 
    f1_score_macro = f1_score(y_true, y_predicted, average='macro') 
    f1_score_binary = f1_score(y_true, y_predicted, average='binary') 
        
    precision = precision_score(y_true, y_predicted, average='micro') 
    recall = recall_score(y_true, y_predicted, average='micro')
    accuracy = accuracy_score(y_true, y_predicted)  
    
    scores = {'f1_score_micro': round(f1_score_micro,2), 
              'f1_score_macro': round(f1_score_macro,2),
              'f1_score_binary': round(f1_score_binary,2),
              'precision' : round(precision,2),
              'recall' : round(recall,2),
              'accuracy' : round(accuracy,2)
              }
    
    return scores

   
def calscore(result,predict):
    fscore=f1_score(result,predict, average='micro')
    accuracy=accuracy_score(result,predict)
    precision=precision_score(result,predict, average='micro')
    recall=recall_score(result,predict, average='micro')
    
    
    return fscore,accuracy,precision,recall


def read_Abdoolahi_data():
    dest_file = r"C:\f5_0_10_no_col.csv"
    
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        data = [data for data in data_iter]
    
    
    #print(data)
    numpy_result = np.asarray(data, dtype = np.int)
   
    _ , cols = numpy_result.shape
    column_names = []

    for names in range(1, cols):
        column_names.append('c' + str(names))
    column_names.append("Person")  
       
    panda_result = pd.DataFrame(data=numpy_result , columns= column_names , dtype = np.int) 
    
    #print(panda_result.columns)
    #print(panda_result)
    
    return panda_result
    
    #return numpy_result

   
    
if __name__ == "__main__":
    
    data = read_Abdoolahi_data()
    #data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), 
    #                    columns=['A', 'B', 'C', 'D', 'res'])
    
    #print(data)
    
    rows , cols = data.shape
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print(cols)
    
    data.iloc[0 , 1] = 3
    data.iloc[1 , 1] = 50

    
    data.iloc[700 , 1] = 50
    data.iloc[701 , 1] = 51
    data.iloc[702 , 1] = 52

    data = shift_each_column_separately(data)
    
    
    train = data[0:700] 
    #print(train)
    test = data[700:]
    
    are_different , data2 = check_data(train , test , remove_latent_variables = True) 
    print(are_different , np.shape(data2))
    if are_different:
        test = data2
    
    resultlist = test['Person'].values
    test = test.drop('Person', axis=1, inplace=False)

    model , scores ,  trainend, testend = bic(train = train, test = test, scoring_function=BicScore, resultlist = resultlist)
    
    print(scores)
