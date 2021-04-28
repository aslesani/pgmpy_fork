# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2020

@author: Adele
"""

import sys
path_of_pycasas_data_dir = r"C:\Users\Adele\Desktop\ref of sMRT\pycasas\pycasas\pycasas\data"
sys.path.append(path_of_pycasas_data_dir)
from _data import CASASDataset

from create_sensor_ids import create_header_string
import numpy as np
import pandas as pd
from datetime import datetime

import inspect

Tm004_address_file = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\tm004_161219_9d"
Kyoto_address_file = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\kyoto_090205_3d"

def get_sensor_status_value(status):
    
    """
    if sensor status is 'ON' or 'OPEN' or 'PRESENT' returns 1,
    if sneosr status is 'OFF' or 'CLOSE' or 'ABSENT' returns 0
    """
    
    if status in ['ON','OPEN', 'PRESENT']:
        return True
        
    elif status in ['OFF','CLOSE','ABSENT']:
        return False
        
    else:
        raise ValueError("the status must be 'ON','OPEN', 'OFF','CLOSE', 'PRESENT' or 'ABSENT'")

def convert_SMRT_datasets_to_each_row_one_features_is_one(dataset_name, address_to_read):
    
    """
    A silly function to convert a reasonable data format to a data format that each row has an on feature, 
    in order to speed up the implementation time due to backward compatibility :))
    Maybe someday I refactor it. Do not promise :D
    """
    dataset = CASASDataset(address_to_read)
    dataset.load_events(show_progress = False) # save events in self.event_list list
    
    dataset._get_sensor_indices() # create the self.sensor_indices_dict which contains the list of sensors with corresponding indices.
    list_of_sensors = sorted(list(dataset.sensor_indices_dict.keys()))
    
    header_string = create_header_string(list_of_sensors)
    header_list = header_string.split(',')
    resident_indices = dataset._get_resident_indices()
    activity_indices = dataset._get_activity_indices()

    # header_list contains all on and off features + person + work. 
    #3 extra columns are needed for date, time and datetime.
    # datetime column is for time ordering and at end, it is removed.   
    number_of_columns = len(header_list) + 3 
    features = [0]* number_of_columns 
    all_features = np.zeros((len(dataset.event_list), number_of_columns), dtype= object )
    
    NaN = float("NaN")
    counter = 0
    
    for event in dataset.event_list:
    
        status_value = get_sensor_status_value(event['message'])
        if status_value == True:
            active_feature = event['sensor'] + '_on'
        elif status_value == False:
            active_feature = event['sensor'] + '_off'
        
        changed_index = header_list.index(active_feature)
        features[changed_index] = 1
        features[-5] = [resident_indices[r] for r in event['resident']]
        features[-4] = [activity_indices[a] for a in event['activity']]
        features[-3] = event['datetime'].strftime("%Y-%m-%d")#date
        features[-2] = event['datetime'].strftime("%H:%M:%S.%f")#time
        features[-1] = event['datetime']
        
        all_features[counter] = features
        counter = counter + 1
        features[changed_index] = 0 # reset the sensor status :D

    print('counter:', counter)
    current_function_name = inspect.currentframe().f_code.co_name
    if counter == len(dataset.event_list):
        print(current_function_name, "Successfully done!")
    else:
        print(inspect.currentframe().f_code.co_name, 'has a mistake!')
        

    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    
    address_to_save = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
    #np.savetxt(address_to_save, np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')
    pd.DataFrame(np.delete(all_features, -1 , 1 )).to_csv(address_to_save, header = None,index = None, )

def read_csv_file(address_to_read):

    pd.read_csv(address_to_read, header = None, index = None)

if __name__ == '__main__':
    #convert_SMRT_datasets_to_each_row_one_features_is_one('Kyoto', Kyoto_address_file)
    dataset_name = 'Kyoto'
    address_to_save_each_row_one_feature_is_on = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
    a = pd.read_csv(address_to_save_each_row_one_feature_is_on, header = None, index_col = None)
    print(a.shape)
    print(type(a.iloc[0]))
    for i in range(20):
        print(a.at[i,141])

    
    