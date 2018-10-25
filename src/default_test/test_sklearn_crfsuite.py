'''
Created on July 21, 2018

@author: Adele
'''
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from read_write import read_sequence_based_CSV_file_with_activity, repaet_person_tags_as_much_as_seq_length

import numpy as np
from pomegranate_test import unison_shuffled_copies

def convert_list_of_int_to_string(data):
    
    for i in range(len(data)):
        data[i] = str(data[i])
    
    return data

def convert_list_of_features_to_dict(data):
    
    data2 = []
    for i in range(len(data)):
        new_row = []
        for j in range(len(data[i])):
            new_row.append({'features': data[i][j]})
        
        data2.append(new_row.copy())
    return data2
        

def apply_CRF_on_data(shuffle):
    
    file_address = r"E:\pgmpy\Seq of sensor events_based on activities\based_on_activities.csv"
    list_of_data , list_of_persons , _ = read_sequence_based_CSV_file_with_activity(file_address = file_address, has_header = True , separate_data_based_on_persons = False)
    
    #print(type(list_of_data[0]) , type(list_of_persons[0]))
    if shuffle:
        list_of_data , list_of_persons = unison_shuffled_copies(list_of_data , list_of_persons)
    
    print(list_of_data[0])

    
    list_of_data = convert_list_of_features_to_dict(list_of_data)
    list_of_persons = convert_list_of_int_to_string(list_of_persons)
    
    #print(list_of_data[0])
    #print("*************")
    #print(list_of_persons[0][0])

    list_of_persons = repaet_person_tags_as_much_as_seq_length(list_of_data,list_of_persons,True)
    
    
    treshhold = int(0.8 * len(list_of_data))
    X_train = list_of_data[0:treshhold]
    y_train = list_of_persons[0:treshhold]
    X_test = list_of_data[treshhold:]
    y_test = list_of_persons[treshhold:]

    #training
    crf = sklearn_crfsuite.CRF(
                               algorithm='lbfgs', 
                               c1=0.1, 
                               c2=0.1, 
                               max_iterations=100, 
                               all_possible_transitions=True
                               )
    
    #print((X_train[0]) , (y_train[0]))
    crf.fit(X_train, y_train)
    
    ## Evaluation
    y_pred = crf.predict(X_test)
    scores = metrics.flat_f1_score(y_test, y_pred, average='weighted')#, labels=labels)
    print(scores)
    
    
if __name__ == "__main__":
    apply_CRF_on_data(shuffle=True)
    
 