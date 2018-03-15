'''
Created on Feb 23, 2018

@author: cloud
'''
import csv
import numpy as np
import pandas as pd
from default_test.DimensionReductionandBNStructureLearning import read_data_from_PCA_digitized_file
from collections import Counter
import collections


def count_one_feature_numbers(dest,feature_column_number ):
    '''
    
    Parameters:
    ===============
    
    '''
    f = open( dest,"r")
    
    all_features = np.zeros((130337, 1), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        cells = line.split(',')
        
        selected_column = cells[feature_column_number] #  is date
        
        
        counter+=1
               
        if first == True:
            first  = False
           
      
        if counter < 130337:
            all_features[counter] = selected_column
        else:
            all_features = np.vstack([all_features,selected_column])
    
    
    return len(set(all_features[:,0]))

def get_group_statistics(dest,data_rows ,feature_column_numbers,feature_column_names, feature_for_grouping ): 
    
    f = open( dest,"r")
    
    all_features = np.zeros((data_rows, 2), dtype= object )
        
    counter = -1
    
    for line in f:
        cells = line.split(',')
        
        selected_column = [cells[i] for i in feature_column_numbers] #  is date
        
        counter+=1
        
        if counter < data_rows:
            all_features[counter] = selected_column
        else:
            all_features = np.vstack([all_features,selected_column])
    

    data = pd.DataFrame(all_features , columns = feature_column_names)
    
    print(data.groupby(feature_for_grouping).agg(['count']))

def get_set_of_features_in_each_column(file_address, number_of_columns):
    
    data = read_data_from_PCA_digitized_file(file_address)
    
    for i in range(0, number_of_columns):
        print("column number: ", i )
        print(collections.Counter(data[: , i]))

               
if __name__ == "__main__":
    #dest = r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"
    dest = r"C:\pgmpy\separation of train and test\31_3\PCA on bag of sensor events_based on activity\train\digitize_bin_200\PCA_n={}.csv"
    #num = count_one_feature_numbers(dest = r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv", feature_column_number = -2)
    #print(num)
    for i in range(2,41):
        #for 
        #get_group_statistics(dest.format(i) ,3056 ,feature_column_numbers=[-2,0], feature_column_names=['date', 'alaki'], feature_for_grouping=['date'])
        print("n=" , i)
        get_set_of_features_in_each_column(dest.format(i), number_of_columns = i )