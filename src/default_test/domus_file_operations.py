'''
Created on Jul 3, 2018

@author: Adele
'''
import mmap
from os import listdir
from os.path import isfile, join
import os.path

import numpy as np
from datetime import timedelta
import time
import datetime
from datetime import timedelta
from create_sensor_ids import create_header_string

from DimensionReductionandBNStructureLearning import PCA_data_generation

file_header_of_all_24_person_sensor_events = "fffe1432adaa01_on,fffe1432adaa01_off,fffe1687d4f801_on,fffe1687d4f801_off,fffe17bab4ba01_on,fffe17bab4ba01_off,fffe1ec74cb801_on,fffe1ec74cb801_off,fffe242ba9aa01_on,fffe242ba9aa01_off,fffe24afb6ab01_on,fffe24afb6ab01_off,fffe28b454a601_on,fffe28b454a601_off,fffe29aadc8c01_on,fffe29aadc8c01_off,fffe29caa5b101_on,fffe29caa5b101_off,fffe29ccbab701_on,fffe29ccbab701_off,fffe2b37eab901_on,fffe2b37eab901_off,fffe3454688901_on,fffe3454688901_off,fffe364a394b01_on,fffe364a394b01_off,fffe3b6e8d5301_on,fffe3b6e8d5301_off,fffe3c8abcc901_on,fffe3c8abcc901_off,fffe3ccc968b01_on,fffe3ccc968b01_off,fffe3dbb8eb801_on,fffe3dbb8eb801_off,fffe43b5ceaa01_on,fffe43b5ceaa01_off,fffe45277dab01_on,fffe45277dab01_off,fffe45ade9ab01_on,fffe45ade9ab01_off,fffe4687b77601_on,fffe4687b77601_off,fffe47a6da8d01_on,fffe47a6da8d01_off,fffe49deac7301_on,fffe49deac7301_off,fffe4ac7547501_on,fffe4ac7547501_off,fffe4ad8adb301_on,fffe4ad8adb301_off,fffe4caa5a4701_on,fffe4caa5a4701_off,fffe571abcec01_on,fffe571abcec01_off,fffe58c9c57501_on,fffe58c9c57501_off,fffe58ca5bbb01_on,fffe58ca5bbb01_off,fffe59ba9bbc01_on,fffe59ba9bbc01_off,fffe5abc547801_on,fffe5abc547801_off,fffe61dc66ba01_on,fffe61dc66ba01_off,fffe63c3a7b201_on,fffe63c3a7b201_off,fffe67daa0c301_on,fffe67daa0c301_off,fffe6ad7aab501_on,fffe6ad7aab501_off,fffe6c8dbc7301_on,fffe6c8dbc7301_off,fffe6d8d7b4a01_on,fffe6d8d7b4a01_off,fffe77a13d5a01_on,fffe77a13d5a01_off,fffe793bf9a301_on,fffe793bf9a301_off,fffe79b6b39c01_on,fffe79b6b39c01_off,fffe7a8b0da901_on,fffe7a8b0da901_off,fffe7aab3aa501_on,fffe7aab3aa501_off,fffe7adb767601_on,fffe7adb767601_off,fffe83b3437801_on,fffe83b3437801_off,fffe8658caaa01_on,fffe8658caaa01_off,fffe886299ab01_on,fffe886299ab01_off,fffe891deb9901_on,fffe891deb9901_off,fffe895b9b9a01_on,fffe895b9b9a01_off,fffe896ad9a901_on,fffe896ad9a901_off,fffe8a49aaab01_on,fffe8a49aaab01_off,fffe8a9cb14b01_on,fffe8a9cb14b01_off,fffe8cc9bad001_on,fffe8cc9bad001_off,fffe8d5b319901_on,fffe8d5b319901_off,fffe8ea1e6dd01_on,fffe8ea1e6dd01_off,fffe919b82b301_on,fffe919b82b301_off,fffe926cca8701_on,fffe926cca8701_off,fffe968db9a601_on,fffe968db9a601_off,fffe9793adaa01_on,fffe9793adaa01_off,fffe988b79ac01_on,fffe988b79ac01_off,fffe9895c7aa01_on,fffe9895c7aa01_off,fffe98d7daa801_on,fffe98d7daa801_off,fffe9992f78d01_on,fffe9992f78d01_off,fffe99b9ab8c01_on,fffe99b9ab8c01_off,fffe9acb3aa401_on,fffe9acb3aa401_off,fffe9b398b4b01_on,fffe9b398b4b01_off,fffe9b5b62c801_on,fffe9b5b62c801_off,fffe9bb67f9601_on,fffe9bb67f9601_off,fffe9c5ebcd801_on,fffe9c5ebcd801_off,fffe9c64a3b901_on,fffe9c64a3b901_off,fffe9d98ab9d01_on,fffe9d98ab9d01_off,fffe9e7ea9a601_on,fffe9e7ea9a601_off,fffea3b8b5a501_on,fffea3b8b5a501_off,fffea4acfdcc01_on,fffea4acfdcc01_off,fffea557aa8b01_on,fffea557aa8b01_off,fffea6a83dae01_on,fffea6a83dae01_off,fffea6d5b96b01_on,fffea6d5b96b01_off,fffea79a5ad901_on,fffea79a5ad901_off,fffea8bcda9c01_on,fffea8bcda9c01_off,fffeaa3de84901_on,fffeaa3de84901_off,fffeaa5ace9d01_on,fffeaa5ace9d01_off,fffeaa6a458e01_on,fffeaa6a458e01_off,fffeaaa3cb8801_on,fffeaaa3cb8801_off,fffeab498e8a01_on,fffeab498e8a01_off,fffeabb51aca01_on,fffeabb51aca01_off,fffeabd88c5301_on,fffeabd88c5301_off,fffeabe38c8a01_on,fffeabe38c8a01_off,fffeac6e75bd01_on,fffeac6e75bd01_off,fffeaca9a34d01_on,fffeaca9a34d01_off,fffeacadcca801_on,fffeacadcca801_off,fffeacea6acb01_on,fffeacea6acb01_off,fffeaf836bda01_on,fffeaf836bda01_off,fffeafe4949d01_on,fffeafe4949d01_off,fffeb2bc639b01_on,fffeb2bc639b01_off,fffeb398ae9601_on,fffeb398ae9601_off,fffeb595a7b101_on,fffeb595a7b101_off,fffeb67c897301_on,fffeb67c897301_off,fffeb6a85dbb01_on,fffeb6a85dbb01_off,fffeb7067da901_on,fffeb7067da901_off,fffeb7f8a5aa01_on,fffeb7f8a5aa01_off,fffeb853ba7601_on,fffeb853ba7601_off,fffeb88797bc01_on,fffeb88797bc01_off,fffeb8a7b9b401_on,fffeb8a7b9b401_off,fffeb8a8da6a01_on,fffeb8a8da6a01_off,fffeb98db7ab01_on,fffeb98db7ab01_off,fffeb99baa8801_on,fffeb99baa8801_off,fffeb9ac9cba01_on,fffeb9ac9cba01_off,fffeb9b1ce8301_on,fffeb9b1ce8301_off,fffeba8386c301_on,fffeba8386c301_off,fffeba97658601_on,fffeba97658601_off,fffeba99abcf01_on,fffeba99abcf01_off,fffebb9b539b01_on,fffebb9b539b01_off,fffebd28b98b01_on,fffebd28b98b01_off,fffebdcf7e6b01_on,fffebdcf7e6b01_off,fffebeae7c8701_on,fffebeae7c8701_off,fffec2a9abba01_on,fffec2a9abba01_off,fffec57da97801_on,fffec57da97801_off,fffec659b23d01_on,fffec659b23d01_off,fffec6a4e7be01_on,fffec6a4e7be01_off,fffec7396cb701_on,fffec7396cb701_off,fffeca59c63901_on,fffeca59c63901_off,fffecaaa7d8f01_on,fffecaaa7d8f01_off,fffecaba26b601_on,fffecaba26b601_off,fffecacb79ba01_on,fffecacb79ba01_off,fffecb4986bb01_on,fffecb4986bb01_off,fffecb72b68501_on,fffecb72b68501_off,fffecb9d516501_on,fffecb9d516501_off,fffecbb299a801_on,fffecbb299a801_off,fffecc84d66b01_on,fffecc84d66b01_off,fffeccb5be8d01_on,fffeccb5be8d01_off,fffecd86a9bb01_on,fffecd86a9bb01_off,fffecdeba79f01_on,fffecdeba79f01_off,fffed3b59e2901_on,fffed3b59e2901_off,fffed3ddc9e701_on,fffed3ddc9e701_off,fffed3e7eceb01_on,fffed3e7eceb01_off,fffed50b88c701_on,fffed50b88c701_off,fffeda599ab601_on,fffeda599ab601_off,fffeda97bb6901_on,fffeda97bb6901_off,fffedabb4b3b01_on,fffedabb4b3b01_off,fffedc6caa6a01_on,fffedc6caa6a01_off,fffedca8baaa01_on,fffedca8baaa01_off,fffedd8ccabe01_on,fffedd8ccabe01_off,fffede7898b601_on,fffede7898b601_off,fffede9668bd01_on,fffede9668bd01_off,fffee4cdda8b01_on,fffee4cdda8b01_off,fffee5acba9301_on,fffee5acba9301_off,fffeeb8d4eac01_on,fffeeb8d4eac01_off,fffeebd42eb701_on,fffeebd42eb701_off,fffeeca8b7d901_on,fffeeca8b7d901_off,fffeecd5198901_on,fffeecd5198901_off,fffef9746ae501_on,fffef9746ae501_off,fffef9ef7d4801_on,fffef9ef7d4801_off,fffefa34bd4d01_on,fffefa34bd4d01_off,fffefa94abc701_on,fffefa94abc701_off,fffefb6453c901_on,fffefb6453c901_off,Person"

def get_number_of_rows(file_address):
    
    f = open(file_address, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    #print(type(readline))
    while readline():
        lines += 1
    return lines


def get_number_of_columns(file_address):
    
    with open(file_address, 'r') as fin:
        data = fin.read().splitlines(True)
        data = data[0].split(',')
        return len(data)
    
    
    

def get_count_of_binary_sensor_events_of_individual(folder_address):
    
    '''
    the domus dataset have smoe folders, each one is corresponding to one person events
    this function count the number of rows of the files which have binary sensors events(*-status files))
    
    '''
    
    list_of_files = [f for f in listdir(folder_address) if isfile(join(folder_address, f))]
    total_rows = 0
    
    for f in list_of_files:
        
        file_name_str = f.split(sep='-')
        try:
            if (len(file_name_str) > 1) and file_name_str[1] == 'status.csv':
                total_rows += get_number_of_rows(join(folder_address, f))
        except Exception as e:
            print("Exception:" , file_name_str)
            
                
    return total_rows
 
 
def get_count_of_binary_sensor_events_for_all_residents():
    
    domus_db = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\{resident}\\'
    
    total_events = 0
    
    for i in range(1,25):
        
        resident_count_of_events = get_count_of_binary_sensor_events_of_individual(domus_db.format(resident = i))
        print("resident" , i , resident_count_of_events)
        total_events += resident_count_of_events
        
    print("total_events:" , total_events)    
        

def get_list_of_sensor_IDs(list_of_person_IDs):
    
    '''
    return list of sensor IDs of the all files which are for list_of_person_IDs
    only binary sensors are considered (status.csv)
    '''
    domus_db = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\{resident}\\'
    
    set_of_sensor_IDs = set()
    
    for person in list_of_person_IDs:
        folder_address = domus_db.format(resident = person)
        list_of_files = [f for f in listdir(folder_address) if isfile(join(folder_address, f))]
        
        for f in list_of_files:
            file_name_str = f.split(sep='-')
            if len(file_name_str)>1 and file_name_str[1] == 'status.csv':
                set_of_sensor_IDs.add(file_name_str[0])
         
            
    return list(sorted(set_of_sensor_IDs))


def create_domus_binary_sensor_events_each_row_one_feature_is_on(all_features, save_address):
    '''
    features: sensor events, person , time
    '''
    
    list_of_sensor_IDs = get_list_of_sensor_IDs(list(range(1,25))) # very important to consider all sensors
    #print("list_of_sensor_IDs:" , list_of_sensor_IDs)
    
    number_of_columns = len(list_of_sensor_IDs) * 2 + 2 # 1 milliseconds + 1 person 
    #number_of_allowed_samples = get_count_of_binary_sensor_events_of_individual(folder_address)
    
    features = np.zeros(number_of_columns-1) # the time is kept separately
    
     
    #all_features = np.zeros((number_of_allowed_samples, number_of_columns), dtype= object )
    number_of_exceptions = 0
    #print("len(all_features):", len(all_features))
    
    new_features = np.zeros(shape=(len(all_features) , number_of_columns-1) , dtype = int)# the time is kept separately
    time_features = np.zeros(shape=(len(all_features) , 1) , dtype = object)
    
    new_feature_row_index = -1 # the index of new_feature matrix
    
    for row_index in range(len(all_features)):
        
        try:
            feature_column = list_of_sensor_IDs.index(all_features[row_index][0])# the sesnor ID
        except Exception as e:# i.e. the item is not in list
            feature_column = -1
            print("exception!!")
            number_of_exceptions +=1
            
        if feature_column != -1:
            sensor_value = all_features[row_index][1]
            
            if sensor_value == '1':
                changed_index = feature_column*2
            elif sensor_value == '0':
                changed_index = feature_column*2 + 1
            else:
                print("the feature is not in {'0','1'}")
            #print("row_index:" , row_index , "sensor_value:" , sensor_value)
            #set_of_changed_index.add(changed_index)
            features[changed_index] = 1
            #features[-1] = all_features[row_index][-1]
            features[-1] = all_features[row_index][-2]
            
            new_feature_row_index +=1
            
            new_features[new_feature_row_index] = features
            time_features[new_feature_row_index] = all_features[row_index][-1]
            
            features[changed_index] = 0
    
    
    #print("number_of_exceptions:" , number_of_exceptions)
    new_features = new_features[ 0 : new_feature_row_index + 1,:]
    time_features = time_features[ 0 : new_feature_row_index + 1,:]

    np.savetxt(save_address, np.concatenate((new_features , time_features) , axis = 1), delimiter=',' , fmt='%s')

def check_file_reader_to_all_be_binary(file_reader):
    '''
    retrun True if all of the values of sensor events be 0 or 1
    '''
    for line in file_reader:
        
        line = line.split(';')
        if line[1] not in ['0' , '1']:
            return False      
    
    return True        

def combine_all_status_files_time_ordered(folder_address, Person_ID, file_address_to_save_events_time_ordered, file_address_to_save_each_row_one_feature_is_on):
    
    list_of_files = [f for f in listdir(folder_address) if isfile(join(folder_address, f))]
    total_number_of_rows = get_count_of_binary_sensor_events_of_individual(folder_address)
    all_features = np.zeros((total_number_of_rows, 4), dtype= object )# sensor_ID, status,person, time
    
    event_number = -1
    
    for f in list_of_files:
        file_name_str = f.split(sep='-')
        if len(file_name_str) > 1  and file_name_str[1] == 'status.csv':
            f = open(join(folder_address, f),"r")
            line_number = -1
            for line in f:
                #print(line)
                line_number +=1
                line = line.split(';')
                if line[1].split('\n')[0] not in ['0' , '1', '-1']:#ignore all of the features corresponding to this sensor ID
                    event_number -= line_number
                    break
                
                event_number += 1
                all_features[event_number] = np.array([file_name_str[0] , line[1].split('\n')[0] ,Person_ID ,line[0]])
              
                #in case of shutter which is binary, with values of -1 and 1, the -1 is converted to 0
                if all_features[event_number][1] == '-1':
                    all_features[event_number][1] = '0'
                    
                        
    print("event_number:", event_number)
    all_features = all_features[0: event_number + 1 , :]
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    
    np.savetxt(file_address_to_save_events_time_ordered, all_features, delimiter=',' , fmt='%s')
    
    print(set(all_features[: , 1]))
    create_domus_binary_sensor_events_each_row_one_feature_is_on(all_features = all_features, save_address = file_address_to_save_each_row_one_feature_is_on )
   

def convert_epoch_time_to_datetime(epoch_time):
    
    '''
    convert epoch time to datetime.
    
    Parameters:
    ===========
    string epoch_time: the time in milli seconds from 1970, jan, 1
    
    Return:
    =======
    datetime
    
    '''
    s, ms = divmod(int(epoch_time), 1000)
    '%s.%03d' % (time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(s)), ms)
    date_time_str = '{}.{:03d}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(s)), ms)
    
    datetime_obj = datetime.datetime.strptime(date_time_str,"%Y-%m-%d %H:%M:%S.%f") 
    #print(datetime_obj)
    return datetime_obj

    
    

def domus_create_bag_of_sensor_events_no_overlap(deltaInMinutes, address_to_read, address_for_save, isSave):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Time in ms from 1.1.1970 00:00
                                          
    2. each row is constructed by considering the bag of sensor events, but no overlap exists in deltaT
    3. in domus we create bag of events features separately.
    
    Parameters:
    ===============
    deltaInMinutes:
    isSave: Save the returned value to file
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    #f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    number_of_entire_rows  = get_number_of_rows(address_to_read)
    number_of_entire_cols = get_number_of_columns(address_to_read)
    
    f = open(address_to_read , "r")
    all_features = np.zeros((number_of_entire_rows, number_of_entire_cols - 1), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        cells = line.split(',')
        #cells[125] = cells[125].split('\n')[0]# the 0th element is the time, the next is empty
        
        converted_cells = [int(i) for i in cells[0: number_of_entire_cols-2]]  
         
        converted_cells.append(convert_epoch_time_to_datetime(cells[-1]))#  is time
        
        counter+=1
           
        if first == True:
            first  = False
            print(len(converted_cells)) 
            #print(converted_cells[-1] , type(converted_cells[-1]))
            #convert_epoch_time_to_datetime(converted_cells[-1])
            Person_ID = int(cells[-2]) # person ID is saved separately    
       
      
        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        


    new_counter = 0
    each_line = 0
    #initialize 
    # create a ndarray with size of all data of the person 
    person_bag = np.ndarray(shape = (number_of_entire_rows , number_of_entire_cols - 2), dtype = int )
    #print(person_bag[each_person].shape)
    for i in range(number_of_entire_cols - 2):
        person_bag[0][i] = all_features[0][i]
    
    
    for offset in range(1, len(all_features), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
            
        timedelta_in_minutes = timedelta.total_seconds(all_features[offset][-1] - all_features[each_line][-1]) / 60
        # compare delta time in minutes
        if timedelta_in_minutes <= deltaInMinutes:
            for i in range(number_of_entire_cols - 2):# 120 is num of features
                person_bag[new_counter][i] += all_features[offset][i]

        else:  
            each_line = offset
            new_counter += 1
            for i in range(number_of_entire_cols - 2):
                person_bag[new_counter][i] = all_features[each_line][i]
            
    #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
    person_bag = np.delete(person_bag , range(new_counter + 1 , number_of_entire_rows + 1 ) , axis = 0)    
         
    #add person_ID column to data
    rows , _ = np.shape(person_bag)
    person_col = np.full( (rows, 1) , Person_ID , dtype = int)  
    person_bag = np.concatenate((person_bag , person_col) , axis = 1) 
    
    if isSave == True:
        #sensor_IDs = get_list_of_sensor_IDs(list(range(1,25)))
        #file_header = create_header_string(a = sensor_IDs, add_work_col = False)
        #print(file_header)
        np.savetxt(address_for_save, person_bag , delimiter=',' , fmt='%s' , header = file_header_of_all_24_person_sensor_events)
    
    
    return person_bag

def convert_list_of_person_IDs_to_string(list_of_Person_IDs):
    
    return str(list_of_Person_IDs).replace('[' , '').replace(']' , '').replace(' ' , '')
   
    
def combine_bag_of_events_data_of_different_people(list_of_Person_IDs):
   
    address_to_read = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\{person}\delta_{delta}min.csv"
    address_to_write = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\combine_person_files_based_on_deltas\{list_of_persons}\delta_{delta}min.csv"
    
    str_list_of_persons = convert_list_of_person_IDs_to_string(list_of_Person_IDs)
    for delta in range(1, 62 , 1):
        
        w_path = address_to_write.format(delta = delta , list_of_persons = str_list_of_persons )
        if not os.path.exists(os.path.dirname(w_path)):
            os.mkdir(os.path.dirname(w_path))
        
        with open(w_path, 'w') as outfile:
            
            first_person = True
            
            for Person_ID in list_of_Person_IDs:
                with open(address_to_read.format(person = Person_ID , delta = delta)) as infile:
                    if first_person:
                        first_person = False
                        outfile.writelines(infile.read().splitlines(True))
                    else:
                        outfile.writelines(infile.read().splitlines(True)[1:])
                    
                              
def create_bag_of_events_for_different_persons_and_deltas():        
    
    address_to_read = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\{person}\all_binary_events_each_row_one_feature_is_on.csv"
    address_to_save = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\{person}\delta_{delta}min.csv"
   
    for Person_ID in range(1,25):
        for delta in range(1, 62 , 1):
            domus_create_bag_of_sensor_events_no_overlap(deltaInMinutes = delta, address_to_read = address_to_read.format(person = Person_ID), address_for_save = address_to_save.format(person = Person_ID , delta = delta), isSave = True)
    
def create_PCA_for_different_bag_of_sensor_events_no_overlap_no_separation(list_of_person_IDS):
    
    for delta in range(1,62,1):#[15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]:#15,30,45,60
        
        directory_for_save = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\PCA on Bag of sensor events_no overlap' + '\\' + convert_list_of_person_IDs_to_string(list_of_person_IDS) + '\\delta=' + str(delta)

        for directory in [directory_for_save]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        directory_for_save = directory_for_save + '\\'
        
        file_address = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\combine_person_files_based_on_deltas\{list_of_persons}\delta_{delta}min.csv'.format(list_of_persons = convert_list_of_person_IDs_to_string(list_of_person_IDS), delta = delta)


        PCA_data_generation(file_address = file_address , base_address_to_save = directory_for_save , remove_date_and_time = False , remove_activity_column = False , has_header = True)


if __name__ == "__main__":
    
    list_of_person_IDS = list(range(1,25))
    #create_bag_of_events_for_different_persons_and_deltas()
    #combine_bag_of_events_data_of_different_people(list(range(1,25)))
    create_PCA_for_different_bag_of_sensor_events_no_overlap_no_separation(list_of_person_IDS)
   
    person = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\{person}\\'
    
    db1 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\1\all_binary_events_each_row_one_feature_is_on.csv"
    #convert_epoch_time_to_datetime('1300953778421')
    #convert_epoch_time_to_datetime('1300953778765')
    #convert_epoch_time_to_datetime('1300953781250')
    
    #domus_create_bag_of_sensor_events_no_overlap(5, address_to_read = db1, address_for_save = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\Bag of sensor events_no overlap_based on different deltas\1\delta_5min.csv", isSave = True)
    #a = get_number_of_columns(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\1\all_binary_events_each_row_one_feature_is_on.csv")
    #print(a)
        
    '''
    for Person_ID in range(1,25):
        #Person_ID = 1
        print("Person_ID:" , Person_ID)
        my_person = person.format(person = Person_ID)
        #get_count_of_binary_sensor_events_of_individual(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\Domus\Dataset\1\\') 
        #get_count_of_binary_sensor_events_for_all_residents()
        #a = get_list_of_sensor_IDs(list(range(1,25)))
        #print(a)
        #print(len(a))
        combine_all_status_files_time_ordered(folder_address = my_person, Person_ID = Person_ID, 
                                              file_address_to_save_events_time_ordered=my_person + "all_binary_events_time_ordered.csv", 
                                              file_address_to_save_each_row_one_feature_is_on= my_person+ "all_binary_events_each_row_one_feature_is_on.csv")
    '''