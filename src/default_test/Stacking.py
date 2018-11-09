'''
Created on Oct 11, 2018

@author: Adele
'''
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from DimensionReductionandBNStructureLearning import digitize_dataset, shift_each_column_separately
from DimensionReductionandBNStructureLearning import shift_2_data_set_based_on_the_first_dataset
from read_write import read_sequence_based_CSV_file_without_activity
from Abdollahi import bic
from Validation import convert_numpy_dataset_to_pandas
from pgmpy.estimators import BicScore
from read_write import separate_dataset_based_on_persons
from pomegranate_test import create_casas7_markov_chain_with_prepared_train_and_test

def BoE_data_preparation(delta, n):
    
    data_address = r"E:\pgmpy\PCA on Bag of sensor events_no overlap\delta={delta}\PCA_n={n}.csv"
    data = digitize_dataset(data_address = data_address.format(delta = delta, n = n), selected_bin = 10, address_to_save = "", isSave=False , has_header = False , return_as_pandas_data_frame = False)
    #print("before shift:", np.shape(data))
    data = shift_each_column_separately(data, False)
    #print("after shift:", np.shape(data))

    list_of_persons = data[:, n:n+1]
    data = data[:, 0:n]
    
    return data, list_of_persons

def SoE_data_preparation(delta):
    
    address_to_read = r"E:\pgmpy\Seq of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
   
    list_of_data , list_of_persons = read_sequence_based_CSV_file_without_activity(
                                        file_address = address_to_read.format(delta= delta), 
                                        has_header = True, 
                                        separate_data_based_on_persons = False)
        
    list_of_data = np.array(list_of_data)
    list_of_persons = np.array(list_of_persons)
   
    return list_of_data, list_of_persons

def BoE_model_building(train_data, train_persons, test_data, test_persons):
    '''
    return the predicted values of test set
    '''
    train_whole = np.concatenate((train_data, train_persons), axis=1)
    test_whole = np.concatenate((test_data, test_persons), axis=1)
    
    train_whole , test_whole, are_different, deleted_indexes = shift_2_data_set_based_on_the_first_dataset(train_whole , test_whole , shiftLastColumn = False)
    
    train_final = convert_numpy_dataset_to_pandas(data = train_whole , addPersonColumn=True)
    test_final = convert_numpy_dataset_to_pandas(data = test_whole , addPersonColumn= True)
    
    test_persons = test_final['Person'].values
    test_data = test_final.drop('Person', axis=1, inplace=False)
    _ , score, _ , _ , predictedValues = bic(train = train_final,test = test_data, scoring_function = BicScore , resultlist = test_persons)
    
    return score , np.array(predictedValues), are_different, deleted_indexes


def SoE_model_building(train_data, train_persons, test_data, test_persons):
  
    train_data, train_persons = separate_dataset_based_on_persons(train_data, 
                                                                  train_persons,
                                                                  0,
                                                                  False)
    
    test_data, test_persons = separate_dataset_based_on_persons(test_data, 
                                                                test_persons,
                                                                0,
                                                                False)
    
    #print(test_persons)
    #print(np.shape(test_persons[0]) , np.shape(test_persons[1]))
    
    
    score, predictedValues = create_casas7_markov_chain_with_prepared_train_and_test(train_set = train_data, 
                                                                                     list_of_persons_in_train=train_persons, 
                                                                                     test_set=test_data, 
                                                                                     list_of_persons_in_test=test_persons, 
                                                                                     return_predicted_lables = True, 
                                                                                     remove_test_columns_which_are_not_in_final_model = True)
    
    return score , predictedValues


def get_avg_f_score(whole_scores):
    
    '''
    calculate f-score for each column separately
    '''
    row , col = np.shape(whole_scores)
    avg_f_score = np.zeros(col , dtype = np.float)
    
    for c in range(col):
        for r in range(row):
            avg_f_score[c] += whole_scores[r][c].get('f1_score_micro')
        
        avg_f_score[c] /= row
            
    print(avg_f_score)
    return avg_f_score


def prepare_data_for_stacking(k, shuffle, delta , n):
    
    BoE_data, BoE_persons = BoE_data_preparation(delta = delta, n = n)
    SoE_data, SoE_persons = SoE_data_preparation(delta=delta)
    
    BoE_predicted_labels = np.zeros(0 , dtype = int)
    SoE_predicted_labels = np.zeros(0 , dtype = int)
    actual_labels = np.zeros(0 , dtype = int)

    #split data to k gruop
    kf = KFold(n_splits=k, random_state=None, shuffle=shuffle)
    k_iter = -1
    whole_scores = np.ndarray(shape = (k, 2) , dtype = dict) # 2 because the sequce-based and bag-based approaches
    for train_index, test_index in kf.split(BoE_data):
       k_iter +=1
       print("k:" , k_iter)
       
       BoE_train_data = BoE_data[train_index]
       BoE_test_data = BoE_data[test_index]
       BoE_train_persons = BoE_persons[train_index]
       BoE_test_persons = BoE_persons[test_index]  
       
       print("len(BoE_test_data) before test:" , len(BoE_test_data))
       score , predictedValues, are_different, deleted_indexes  = BoE_model_building(BoE_train_data, BoE_train_persons, BoE_test_data, BoE_test_persons)
       whole_scores[k_iter][0] = score
       #print("#####", len(predictedValues))
       #print("BOE:" , score.get('f1_score_micro'))
       BoE_predicted_labels = np.concatenate((BoE_predicted_labels , predictedValues) , axis = 0)
       print("len(BoE_test_data) afrter test:" , len(predictedValues))

       
       SoE_train_data = SoE_data[train_index]
       SoE_test_data = SoE_data[test_index]
       SoE_train_persons = SoE_persons[train_index]
       SoE_test_persons = SoE_persons[test_index]
       
       # if some of the rows in the BoE_test are removed, 
       #so the corresponding rows are removed from the SoE_test_data as well.
       if are_different:
           SoE_test_data = np.delete(SoE_test_data , deleted_indexes , axis = 0)
           SoE_test_persons = np.delete(SoE_test_persons, deleted_indexes , axis = 0)

       print("len(SoE_test_data):" , len(SoE_test_data))
       score , predictedValues = SoE_model_building(SoE_train_data, SoE_train_persons, SoE_test_data, SoE_test_persons)
       whole_scores[k_iter][1] = score
       #print("SOE:" , score.get('f1_score_micro'))
       SoE_predicted_labels = np.concatenate((SoE_predicted_labels , predictedValues) , axis = 0)
       
       actual_labels = np.concatenate((actual_labels , SoE_test_persons) , axis = 0)

       #print(len(BoE_predicted_labels) , len(SoE_predicted_labels) , len(actual_labels))    


    avg_f_score = get_avg_f_score(whole_scores)#np.mean(scores, axis = 1)
    
    print("******************")
    print(len(BoE_predicted_labels) , len(SoE_predicted_labels) , len(actual_labels))    
    
    #for i in range(len(actual_labels)):
     #   print(BoE_predicted_labels[i] , SoE_predicted_labels[i] , actual_labels[i])
    
    
def test_kf_split():
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    a = list(range(1,11))
    b = [a[i]*10 for i in range(0,len(a))] 
    a = np.array(a , dtype = int)
    b = np.array(b , dtype = int)
    
    
    for ind1,ind2 in kf.split(b):
        #print(b[ind1[0]])
        print('a1:' , a[ind1] , 'a2:' , a[ind2])
        print('b1:' , b[ind1] , 'b2:' , b[ind2])

    
   

if __name__ == "__main__":
    
    prepare_data_for_stacking(k = 10, shuffle = True, delta = 150, n=10)
   
    #test_kf_split()
    #a1, b1 = BoE_data_preparation(1100,2)
    '''a2, b2 = SoE_data_preparation(1100)
    print(len(b1))
    print(len(b2))
    
    for i in range(len(b1)):
        if b1[i] != b2[i]:
            print(i)
    print( set(np.equal(b1, b2)))
    '''
    #print(a)
    #print(set(b))
     
