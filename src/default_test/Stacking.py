'''
Created on Oct 11, 2018

@author: Adele
'''
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from DimensionReductionandBNStructureLearning import digitize_dataset, shift_each_column_separately
from DimensionReductionandBNStructureLearning import shift_2_data_set_based_on_the_first_dataset
from read_write import read_sequence_based_CSV_file_without_activity
from data_utils import check_data
from Abdollahi import bic
from Validation import convert_numpy_dataset_to_pandas
from pgmpy.estimators import BicScore



def BoE_data_preparation(delta, n):
    
    data_address = r"E:\pgmpy\PCA on Bag of sensor events_no overlap\delta={delta}\PCA_n={n}.csv"
    data = digitize_dataset(data_address = data_address.format(delta = delta, n = n), selected_bin = 10, address_to_save = "", isSave=False , has_header = False , return_as_pandas_data_frame = False)
    data = shift_each_column_separately(data, False)

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
    
    train_whole , test_whole = shift_2_data_set_based_on_the_first_dataset(train_whole , test_whole , shiftLastColumn = False)
    
    train_final = convert_numpy_dataset_to_pandas(data = train_whole , addPersonColumn=True)
    test_final = convert_numpy_dataset_to_pandas(data = test_whole , addPersonColumn= True)
    
    test_persons = test_final['Person'].values
    test_data = test_final.drop('Person', axis=1, inplace=False)
  
    _ , score, _ , _ , predictedValues = bic(train = train_final,test = test_data, scoring_function = BicScore , resultlist = test_persons)
    
    print(score , predictedValues)

    

def prepare_data_for_stacking(k, shuffle, delta , n):
    
    BoE_data, BoE_persons = BoE_data_preparation(delta = delta, n = n)
    SoE_data, SoE_persons = SoE_data_preparation(delta=delta)
    
    #split data to k gruop
    kf = KFold(n_splits=k, random_state=None, shuffle=shuffle)
    k_iter = -1
    scores = np.ndarray(shape = (2, k) , dtype = float) # 2 because the sequce-based and bag-based approaches
    for train_index, test_index in kf.split(BoE_data):
       k_iter +=1
       
       BoE_train_data = BoE_data[train_index]
       BoE_test_data = BoE_data[test_index]
       BoE_train_persons = BoE_persons[train_index]
       BoE_test_persons = BoE_persons[test_index]
         
       SoE_train_data = SoE_data[train_index]
       SoE_test_data = SoE_data[test_index]
       SoE_train_persons = SoE_persons[train_index]
       SoE_test_persons = SoE_persons[test_index]
       
       BoE_model_building(BoE_train_data, BoE_train_persons, BoE_test_data, BoE_test_persons)
       break
       #scores[clasifier_index][k_iter]= f1_score(y_true = y_test, y_pred = predicted, average = 'micro')#cross_val_score(clf, data, target, cv=10 , scoring='f1_macro') ####10-fold cross validation 
    
    #avg_f_score = np.mean(scores, axis = 1)
    
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
     
