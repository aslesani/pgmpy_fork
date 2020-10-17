'''
Created on Oct 4, 2020

@author: Adele
=========================


'''

# coding=utf8


from os.path import join
import numpy as np
import csv
from itertools import count
import re
import time
import dateutil
import datetime
from datetime import timedelta
from numpy import dtype
from read_write import read_data_from_CSV_file
import pickle
import pandas as pd


file_header_Twor2009 = "D03_on,D03_off,D05_on,D05_off,D07_on,D07_off,D08_on,D08_off,D09_on,D09_off,D10_on,D10_off,D12_on,D12_off,D14_on,D14_off,D15_on,D15_off,I03_on,I03_off,M01_on,M01_off,M02_on,M02_off,M03_on,M03_off,M04_on,M04_off,M05_on,M05_off,M06_on,M06_off,M07_on,M07_off,M08_on,M08_off,M09_on,M09_off,M10_on,M10_off,M11_on,M11_off,M12_on,M12_off,M13_on,M13_off,M14_on,M14_off,M15_on,M15_off,M16_on,M16_off,M17_on,M17_off,M18_on,M18_off,M19_on,M19_off,M20_on,M20_off,M21_on,M21_off,M22_on,M22_off,M23_on,M23_off,M24_on,M24_off,M25_on,M25_off,M26_on,M26_off,M27_on,M27_off,M28_on,M28_off,M29_on,M29_off,M30_on,M30_off,M31_on,M31_off,M32_on,M32_off,M33_on,M33_off,M34_on,M34_off,M35_on,M35_off,M36_on,M36_off,M37_on,M37_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M42_on,M42_off,M43_on,M43_off,M44_on,M44_off,M45_on,M45_off,M46_on,M46_off,M47_on,M47_off,M48_on,M48_off,M49_on,M49_off,M50_on,M50_off,M51_on,M51_off,Person,Work"
#"M01_on,M01_off,M02_on,M02_off,M03_on,M03_off,M04_on,M04_off,M05_on,M05_off,M06_on,M06_off,M07_on,M07_off,M08_on,M08_off,M09_on,M09_off,M10_on,M10_off,M11_on,M11_off,M12_on,M12_off,M13_on,M13_off,M14_on,M14_off,M15_on,M15_off,M16_on,M16_off,M17_on,M17_off,M18_on,M18_off,M19_on,M19_off,M20_on,M20_off,M21_on,M21_off,M22_on,M22_off,M23_on,M23_off,M24_on,M24_off,M25_on,M25_off,M26_on,M26_off,M27_on,M27_off,M28_on,M28_off,M29_on,M29_off,M30_on,M30_off,M31_on,M31_off,M32_on,M32_off,M33_on,M33_off,M34_on,M34_off,M35_on,M35_off,M36_on,M36_off,M37_on,M37_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M42_on,M42_off,M43_on,M43_off,M44_on,M44_off,M45_on,M45_off,M46_on,M46_off,M47_on,M47_off,M48_on,M48_off,M49_on,M49_off,M50_on,M50_off,M51_on,M51_off,I03_on,I03_off,D03_on,D03_off,D05_on,D05_off,D07_on,D07_off,D08_on,D08_off,D09_on,D09_off,D10_on,D10_off,D12_on,D12_off,D14_on,D14_off,D15_on,D15_off,Person,Work"
file_header_Twor2010 = "D001_on,D001_off,D002_on,D002_off,D003_on,D003_off,D004_on,D004_off,D005_on,D005_off,D006_on,D006_off,D007_on,D007_off,D008_on,D008_off,D009_on,D009_off,D010_on,D010_off,D011_on,D011_off,D012_on,D012_off,D013_on,D013_off,D014_on,D014_off,D015_on,D015_off,I006_on,I006_off,I010_on,I010_off,I011_on,I011_off,I012_on,I012_off,M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,M032_on,M032_off,M033_on,M033_off,M034_on,M034_off,M035_on,M035_off,M036_on,M036_off,M037_on,M037_off,M038_on,M038_off,M039_on,M039_off,M040_on,M040_off,M041_on,M041_off,M042_on,M042_off,M043_on,M043_off,M044_on,M044_off,M045_on,M045_off,M046_on,M046_off,M047_on,M047_off,M048_on,M048_off,M049_on,M049_off,M050_on,M050_off,M051_on,M051_off,Person,Work"
file_header_Tulum2009 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,Person,Work"
file_header_Tulum2010 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,Person,Work"
file_header_OpenSHS = "wardrobe_on,wardrobe_off,tv_on,tv_off,oven_on,oven_off,officeLight_on,officeLight_off,officeDoorLock_on,officeDoorLock_off,officeDoor_on,officeDoor_off,officeCarp_on,officeCarp_off,office_on,office_off,mainDoorLock_on,mainDoorLock_off,mainDoor_on,mainDoor_off,livingLight_on,livingLight_off,livingCarp_on,livingCarp_off,kitchenLight_on,kitchenLight_off,kitchenDoorLock_on,kitchenDoorLock_off,kitchenDoor_on,kitchenDoor_off,kitchenCarp_on,kitchenCarp_off,hallwayLight_on,hallwayLight_off,fridge_on,fridge_off,couch_on,couch_off,bedroomLight_on,bedroomLight_off,bedroomDoorLock_on,bedroomDoorLock_off,bedroomDoor_on,bedroomDoor_off,bedroomCarp_on,bedroomCarp_off,bedTableLamp_on,bedTableLamp_off,bed_on,bed_off,bathroomLight_on,bathroomLight_off,bathroomDoorLock_on,bathroomDoorLock_off,bathroomDoor_on,bathroomDoor_off,bathroomCarp_on,bathroomCarp_off,Person,Work"
file_header_Test = "M32_on,M32_off,M35_on,M35_off,M36_on,M36_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M45_on,M45_off,M47_on,M47_off,M48_on,M48_off,Person,Work"
   

def read_pickle_file_data_as_pandas_dataframe(address):
    
    fp = open(address, 'rb')
    data = pickle.load(fp)
    fp.close()
    
    data = pd.DataFrame.from_dict(data)
    
    return data
    
    
    
def list_of_real_persons_associted_with_sensor_event(data_row, real_person_columns):
    
    next_person_ID = [i for i in real_person_columns if data_row[i] == 1]
    if len(next_person_ID) > 1: # if the event is created by more than one person or nobody
        raise Exception("There is more than or no person assicuated with this sensor event. \
                        We can not process your sensor events in this version of code.\
                        datarow ={} \n".format(data_row))
    elif len(next_person_ID) == 0:# in this version of code we ignore events associated with none of the residents
        return None
    return int(next_person_ID[0].split('R')[1])

def list_of_predicted_persons_associted_with_sensor_event(data_row, predicted_person_columns):
    
    next_person_ID = [i for i in predicted_person_columns if data_row[i] == 1]
    return int(next_person_ID[0].split('R')[1])

def remove_none_person_tags_from_sequence_of_person_tags(data):
    
    '''
    '''
    index = list(range(len(data)))
    index.reverse()# in reverse order, removing a row can not affect on previous rows indexes.
    
    for i in index:
        locs = np.where(np.equal(data[i][1], None))
        if len(locs[0]) == len(data[i][1]): # if all of the items in row is None, delete the row
            data = np.delete(data, i , axis = 0)
        else:
            data[i][0] = np.delete(data[i][0], locs)
            data[i][1] = np.delete(data[i][1], locs)
        
    return data

def create_sequence_of_events_based_on_delta(deltaInMinutes, data,\
                     considered_columns_as_person_tags,return_sequence_of_ground_truth_labels):
    '''
    it is a general method to create sequence of sensor events based on time delta.
 
    Parameters:
    ==========
    deltaInMinutes: 
    data: pnadas DataFrame
    considered_columns_as_person_tags: list of column names to consider as person tags
    return_sequence_of_ground_truth_labels: if True, the function create a sequence of real person tags 
                                            for each sequence.
    Return:
    =======
    
   
    '''
    # create the column tags counterpart
    real_person_columns = [col.split('_')[0] for col in considered_columns_as_person_tags]
    number_of_residents = len(real_person_columns)
    #to save seq of sensor_events for each person separately 
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)

    #create sequences based on considered_columns_as_person_tags
    for each_person in range(number_of_residents):
        #initialize 
        #Note: pandas dataframe considers the number_of_train_samples th data as well.
        person_data = data[data[considered_columns_as_person_tags[each_person]] == 1]
        person_data_number_of_rows , _ = person_data.shape
        new_counter = 0
        current_person_tag =  int(considered_columns_as_person_tags[each_person].split('R')[1].split('_')[0])
        each_line = person_data.index[0]
        
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person 
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 3), dtype = np.ndarray )
        
        first_valid_event = -1
        next_person_ID = None
        while next_person_ID == None:
            first_valid_event = first_valid_event + 1
            next_person_ID = list_of_real_persons_associted_with_sensor_event(\
                                        data_row = person_data.loc[person_data.index[first_valid_event]], 
                                        real_person_columns = real_person_columns)
            
        
        person_sequences[each_person][0][1] = [next_person_ID]
        
        next_element_of_sequence = person_data.loc[person_data.index[first_valid_event]]['sensor']        
        #print(next_element_of_sequence)
        person_sequences[each_person][0][0] = [next_element_of_sequence]
        
        
        for offset in person_data.index[first_valid_event+1:]:
            #print('offset:', offset)    
            timedelta_in_minutes = timedelta.total_seconds(person_data.loc[offset]['start'] - \
                                   person_data.loc[each_line]['start']) / 60
            # compare delta time in minutes
            if timedelta_in_minutes <= deltaInMinutes:
                
                next_person_ID = list_of_real_persons_associted_with_sensor_event(\
                                                    data_row = person_data.loc[offset], 
                                                    real_person_columns = real_person_columns)
                
                person_sequences[each_person][new_counter][1].append(next_person_ID)
                
                next_element_of_sequence = person_data.loc[offset]['sensor']        
                #print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0].append(next_element_of_sequence)
                

            else:  
                person_sequences[each_person][new_counter][2] = current_person_tag
                each_line = offset
                new_counter += 1
                next_person_ID = list_of_real_persons_associted_with_sensor_event(\
                                                    data_row = person_data.loc[offset], 
                                                    real_person_columns = real_person_columns)
                    
                person_sequences[each_person][new_counter][1] = [next_person_ID]
                
                next_element_of_sequence = person_data.loc[offset]['sensor']        
                #print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0] = [next_element_of_sequence]
                
        
        person_sequences[each_person][new_counter][2] = current_person_tag # update the last row person tag
              
                
        #remove additional items (because the initial size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , \
                                                  person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_sequences[0]
    for person in range(1, number_of_residents):
        #print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)
    
    person_sequences[0] = remove_none_person_tags_from_sequence_of_person_tags(person_sequences[0])
    
    if return_sequence_of_ground_truth_labels == False:
        person_sequences[0] = np.delete(person_sequences[0], 1 , axis = 1) # delete column 1 which is sequence of real person tags
    
    return person_sequences[0]


def add_max_frequent_person_tag_in_sequence_of_person_tags_column(data):
    '''
    '''
    number_of_data_rows = len(data)
    data = np.c_[data, np.zeros(number_of_data_rows , dtype = np.int)]
    for i in range(number_of_data_rows):
        data[i,2] = max(data[i,1], key = data[i,1].count)
    
    return data

def create_sequence_of_sensor_events_based_on_delta_no_overlap_train_based_on_real_tags_and_nontrain_based_on_prediction_seprate_train_and_non_train(\
                        deltaInMinutes, address_to_read, isSave, directory_for_save, train_percent):
    '''
    read a pickle file of prediction as pandas dataframe.
    this function separate train and non-train data.
    train data are converted to sequences based on real person tags.
    non-train data are converted to sequences based on predicted person tags.
    
    the columns are: 'start', 
                     'stop', 
                     'Rx': the columns representing the Residetns IDs, if the column is 1, it means that that column is the PID of that event
                     'Rx_predited': the columns representing the predicted IDs.
                     'sensor': the sensor that is activated in event
                     'activity': the performed activity
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    train_percent: percent of data(sensor events) to be considered as train data. 
                   To create sequence of events from train data, the real person tag is considered, 
                   while for validation and test data, the predicted tag is considered. 
                   The real tag of each sensor event of non-train data should be returned for evaluation. 
    
    '''
    data = read_pickle_file_data_as_pandas_dataframe(address_to_read)
    real_person_columns = [i for i in data.columns if re.findall('^R[0-9]{1}$', i)]
    predicted_person_columns = [i for i in data.columns if re.findall('^R[0-9]{1}_predicted$', i)]
    

    rows , cols = data.shape
     
    #seperate each person data based on real tags
    number_of_residents = len(real_person_columns)
    number_of_train_samples = int(train_percent * rows)
    
    train_data = data.head(number_of_train_samples)
    non_train_data = data.tail(rows - number_of_train_samples)
    
    
    train_seq = create_sequence_of_events_based_on_delta(deltaInMinutes = deltaInMinutes, 
                                                         data = train_data, 
                                                         considered_columns_as_person_tags = real_person_columns,
                                                         return_sequence_of_ground_truth_labels = False)
   
    non_train_seq = create_sequence_of_events_based_on_delta(deltaInMinutes = deltaInMinutes, 
                                                             data = non_train_data, 
                                                             considered_columns_as_person_tags = predicted_person_columns,
                                                             return_sequence_of_ground_truth_labels = True)
    

    #non_train_seq = add_max_frequent_person_tag_in_sequence_of_person_tags_column(non_train_seq)
    
    if isSave:
        train_and_nontrain_file_address = join(directory_for_save, \
                                          'trainPercent_{}'.format(train_percent),
                                          '{delta}_min.pkl'.format(delta = deltaInMinutes))
        with open(train_and_nontrain_file_address, 'wb') as f:
            pickle.dump([train_seq, non_train_seq], f)
            print('{} processing saved!'.format(deltaInMinutes))
       

def create_sequence_of_sensor_events_based_on_delta_no_overlap_all_based_on_prediction_seperate_train_and_non_train(\
                                deltaInMinutes, address_to_read, isSave, directory_for_save, train_percent):
    '''
    read a pickle file of prediction as pandas dataframe.
    this function separate train and non-train data.
    all data are converted to sequences based on predicted person tags.
    
    the columns are: 'start', 
                     'stop', 
                     'Rx': the columns representing the Residetns IDs, if the column is 1, it means that that column is the PID of that event
                     'Rx_predited': the columns representing the predicted IDs.
                     'sensor': the sensor that is activated in event
                     'activity': the performed activity
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    train_percent: percent of data(sensor events) to be considered as train data. 
                   To create sequence of events the predicted tag is considered. 
                   The real tag of each sensor event should be returned for evaluation. 
    
    '''
    
    data = read_pickle_file_data_as_pandas_dataframe(address_to_read)
    real_person_columns = [i for i in data.columns if re.findall('^R[0-9]{1}$', i)]
    predicted_person_columns = [i for i in data.columns if re.findall('^R[0-9]{1}_predicted$', i)]
    

    rows , cols = data.shape
     
    #seperate each person data based on real tags
    number_of_residents = len(real_person_columns)
    number_of_train_samples = int(train_percent * rows)
    
    train_data = data.head(number_of_train_samples)
    non_train_data = data.tail(rows - number_of_train_samples)
    
   
    train_seq = create_sequence_of_events_based_on_delta(deltaInMinutes = deltaInMinutes, 
                                                        data = train_data, 
                                                        considered_columns_as_person_tags = predicted_person_columns,
                                                        return_sequence_of_ground_truth_labels = True)
    
    non_train_seq = create_sequence_of_events_based_on_delta(deltaInMinutes = deltaInMinutes, 
                                                        data = non_train_data, 
                                                        considered_columns_as_person_tags = predicted_person_columns,
                                                        return_sequence_of_ground_truth_labels = True)
    
    #train_seq = add_max_frequent_person_tag_in_sequence_of_person_tags_column(train_seq)
    #non_train_seq = add_max_frequent_person_tag_in_sequence_of_person_tags_column(non_train_seq)
    
    if isSave:
        train_and_nontrain_file_address = join(directory_for_save, \
                                          'trainPercent_{}'.format(train_percent),
                                          '{delta}_min.pkl'.format(delta = deltaInMinutes))
        with open(train_and_nontrain_file_address, 'wb') as f:
            pickle.dump([train_seq, non_train_seq], f)
            print('{} processing saved!'.format(deltaInMinutes))
       


if __name__ == '__main__':
            
    file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
    file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
    file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"
    file_address_Twor2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\twor.2010\data_edited_by_adele"
    file_address_Test = r"E:\pgmpy\Test\annotated"
    
    pickle_file = r"E:\pgmpy\{}\just_on_events_with_predicted_person_tags.pkl"
    save_directory_sequence = r"E:\pgmpy\{dataset}\train_and_nontrain_sequences_all_based_on_tracking"
    dataset_name = 'Twor2009'
    #train_percent = 0.6
    
    '''
    data = read_pickle_file_data_as_pandas_dataframe(pickle_file.format(dataset_name))
    data['start'] = 0
    print(data.loc[0])
    '''
    for train_percent in [0.6,0.7,0.8]:
        for deltaInMinutes in  list(range(1,11)) + [15,30,45,60,75,90,100]:
            create_sequence_of_sensor_events_based_on_delta_no_overlap_all_based_on_prediction_seperate_train_and_non_train(\
                       deltaInMinutes = deltaInMinutes, 
                       address_to_read = pickle_file.format(dataset_name), 
                       isSave = True, 
                       directory_for_save = save_directory_sequence.format(dataset = dataset_name), 
                       train_percent = train_percent)
    
    