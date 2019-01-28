'''
Created on May 14, 2017

@author: Adele

========================================
This module provide functions to convert format of casas data. The ones that contain
an event per line are converted to full feature format
========================================


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
from xml.sax.handler import all_features
#from builtins import int
from numpy import dtype
from read_write import read_data_from_CSV_file
#from docutils.parsers import null

#from matplotlib.pyplot import axis
#from nntplib import lines



work_lists = ["0"]

file_header_Twor2009 = "M01_on,M01_off,M02_on,M02_off,M03_on,M03_off,M04_on,M04_off,M05_on,M05_off,M06_on,M06_off,M07_on,M07_off,M08_on,M08_off,M09_on,M09_off,M10_on,M10_off,M11_on,M11_off,M12_on,M12_off,M13_on,M13_off,M14_on,M14_off,M15_on,M15_off,M16_on,M16_off,M17_on,M17_off,M18_on,M18_off,M19_on,M19_off,M20_on,M20_off,M21_on,M21_off,M22_on,M22_off,M23_on,M23_off,M24_on,M24_off,M25_on,M25_off,M26_on,M26_off,M27_on,M27_off,M28_on,M28_off,M29_on,M29_off,M30_on,M30_off,M31_on,M31_off,M32_on,M32_off,M33_on,M33_off,M34_on,M34_off,M35_on,M35_off,M36_on,M36_off,M37_on,M37_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M42_on,M42_off,M43_on,M43_off,M44_on,M44_off,M45_on,M45_off,M46_on,M46_off,M47_on,M47_off,M48_on,M48_off,M49_on,M49_off,M50_on,M50_off,M51_on,M51_off,I03_on,I03_off,D03_on,D03_off,D05_on,D05_off,D07_on,D07_off,D08_on,D08_off,D09_on,D09_off,D10_on,D10_off,D12_on,D12_off,D14_on,D14_off,D15_on,D15_off,Person,Work"
file_header_Twor2010 = "D001_on,D001_off,D002_on,D002_off,D003_on,D003_off,D004_on,D004_off,D005_on,D005_off,D006_on,D006_off,D007_on,D007_off,D008_on,D008_off,D009_on,D009_off,D010_on,D010_off,D011_on,D011_off,D012_on,D012_off,D013_on,D013_off,D014_on,D014_off,D015_on,D015_off,I006_on,I006_off,I010_on,I010_off,I011_on,I011_off,I012_on,I012_off,M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,M032_on,M032_off,M033_on,M033_off,M034_on,M034_off,M035_on,M035_off,M036_on,M036_off,M037_on,M037_off,M038_on,M038_off,M039_on,M039_off,M040_on,M040_off,M041_on,M041_off,M042_on,M042_off,M043_on,M043_off,M044_on,M044_off,M045_on,M045_off,M046_on,M046_off,M047_on,M047_off,M048_on,M048_off,M049_on,M049_off,M050_on,M050_off,M051_on,M051_off,Person,Work"
file_header_Tulum2009 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,Person,Work"
file_header_Tulum2010 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,Person,Work"



def casas7_to_csv():
    '''
    1. annotated file is processed
    2. just motion, item and door sensors are kept (binary sensors),
         others (i.e. burner, water,temprature and electricity usage) are not. 
    3. 'on' converted to 1
        'off' converted to 0
        'open' converted to 1
        'close' converted to 0
        'present' converted to 0 (because the item is not used)
        'absent' converted to 1
        date and time removed
    4. I supposed the default value of sensors is not important, so set them to 0.
    5. The order of sensor features: 51 motion sensors, 
                                     1 item sensor(we have 9 item sensors, but just i03 is used),
                                     9 door sensors (03, 05 , 07, 08, 09, 10 , 12, 14, 15),
                                     Person,
                                     Work 
    6. Wash_bathtub and Cleaning does not have person number! I supposed the R1 as the person (the kolfat :D)
    7. Because I know the final data has 138039 rows. i defined the array so...
    8. the first instance is name of features
    '''
    f = open( join(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"),"r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    features = [0]* 63
    # for month2, the rows are 64466 (+1)
    all_features = np.zeros((130337, 63), dtype= np.int )#np.str)130336 +1
    feature_names = ["M01", "M02", "M03", "M04" , "M05" , "M06" , "M07" , "M08" , "M09" , "M10"
                       , "M11", "M12", "M13", "M14" , "M15" , "M16" , "M17" , "M18" , "M19" , "M20"
                       , "M21", "M22", "M23", "M24" , "M25" , "M26" , "M27" , "M28" , "M29" , "M30"
                       , "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40"
                       , "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50"
                       , "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14"
                       , "D15", "PNo", "WNo"]
    
    #print(feature_names)
    #used_features = []
    counter = -1
    #print(features)
    first = True
    for line in f:
        
        cells = line.split()
        #print(cells)
        feature_column = get_feature_column(cells[2])
        if feature_column != -1:
            counter +=1
            #if counter > 1000:
            #    break
            
            sensor_value = get_sensor_value(cells[3])
            features[feature_column] = sensor_value
           
            
            if len(cells) > 4:
                PersonNumber, WorkNumber = get_person_and_work(cells[4])
                features[-2] = PersonNumber
                features[-1] = WorkNumber
                # i.e. the line is annotated
        
            if first == True:
                first  = False
            
            if counter < 130337:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
      
    
    rows, cols = all_features.shape
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\data.csv', 
               all_features, delimiter=',' , fmt='%s')


def casas7_to_csv_time_Ordered():
    '''
    1. annotated file is processed (for Twor2009)
    2. just motion, item and door sensors are kept (binary sensors),
         others (i.e. burner, water,temprature and electricity usage) are not. 
    3. 'on' converted to 1
        'off' converted to 0
        'open' converted to 1
        'close' converted to 0
        'present' converted to 0 (because the item is not used)
        'absent' converted to 1
        date and time removed
    4. I supposed the default value of sensors is not important, so set them to 0.
    5. The order of sensor features(in Towr 2009): 51 motion sensors, 
                                     1 item sensor(we have 9 item sensors, but just i03 is used),
                                     9 door sensors (03, 05 , 07, 08, 09, 10 , 12, 14, 15),
                                     Person,
                                     Work,
                                     Date,
                                     Time
    6. Wash_bathtub and Cleaning does not have person number! I supposed the R1 as the person (the kolfat :D)
    7. Because I know the final data has 138039 rows. i defined the array so...
    8. the first instance is name of features
    9. Events are labeled with time and date and the data is sorted based on datatime
    '''
    f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated","r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    features = [0]* 66 # 63 features + 1 date + 1 time + 1 datetime for ordering data
    # for month2, the rows are 64466 (+1)
    all_features = np.zeros((130337, 66), dtype= object )#np.str)130336 +1
   
    counter = -1
    #print(features)
    first = True
    for line in f:
        
        cells = line.split()
        #print(cells)
        feature_column = get_feature_column(cells[2])
        if feature_column != -1:
            counter +=1
            sensor_value = get_sensor_value(cells[3])
            features[feature_column] = sensor_value
           
            
            if len(cells) > 4:
                PersonNumber, WorkNumber = get_person_and_work(cells[4])
                features[-5] = PersonNumber
                features[-4] = WorkNumber
                # i.e. the line is annotated
            features[-3] = cells[0]
            features[-2] = cells[1]
            features[-1] = convert_string_to_datetime(cells[0],cells[1])
            
            if first == True:
                first  = False
            
            if counter < 130337:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
    
    
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column

    rows, cols = all_features.shape
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data+time_ordered.csv', 
               np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')


def casas7_to_csv_based_on_sensor_events_time_Ordered(file_address_to_read, file_address_to_save):
    '''
    0. It is important that the difference between this method and casas7_to_csv_time_Ordered
       is that in this method the si-on and si-off are considered as two different features. if they 
       are used, the corresponding feature is set to 1. else 0 .
    1. annotated file is processed
    2. just motion, item and door sensors are kept (binary sensors),
         others (i.e. burner, water,temprature and electricity usage) are not. 
    3. 'on' converted to 1
        'off' converted to 0
        'open' converted to 1
        'close' converted to 0
        'present' converted to 0 (because the item is not used)
        'absent' converted to 1
        date and time removed
    4. I supposed the default value of sensors is not important, so set them to 0.
    5. The order of sensor features (each sensor has 2 features (on and off)): 51 motion sensors, 
                                     1 item sensor(we have 9 item sensors, but just i03 is used),
                                     9 door sensors (03, 05 , 07, 08, 09, 10 , 12, 14, 15),
                                     Person,
                                     Work,
                                     Date,
                                     Time
    6. Wash_bathtub and Cleaning does not have person number! I supposed the R1 as the person (the kolfat :D)
    7. Because I know the final data has 138039 rows. i defined the array so...
    8. the first instance is name of features
    9. Events are labeled with time and date and the data is sorted based on datatime

    Important: in Tulum 2010 some of the first lines which are not taged for a specific person are removed manually. 
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated","r")
    f = open(file_address_to_read,"r")
    list_of_sensors, number_of_allowed_samples, list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(file_address_to_read)
    
    number_of_columns = len(list_of_sensors) * 2 + 5 # 127 for Towr2009
    # 61 * 2 features + Person + work + 1 date + 1 time + 1 datetime for ordering data
    features = [0] * number_of_columns 
    # for month2, the rows are 64466 (+1)
    all_features = np.zeros((number_of_allowed_samples, number_of_columns), dtype= object )#np.str)130336 +1
    
    counter = -1
    first = True
    for line in f:
        
        cells = line.split()
        
        try:
            feature_column = list_of_sensors.index(cells[2])# get_feature_column(cells[2]) is for Twor2009 dataset and is set manually
        except Exception as e:# i.e. the item is not in list
            feature_column = -1
            
            
        if feature_column != -1:
            counter +=1
           
            sensor_value = get_sensor_value(cells[3])
            
            if sensor_value == '1':
                changed_index = feature_column*2
            elif sensor_value == '0':
                #sensor_value == 0 
                changed_index = feature_column*2 + 1
            else:
                print("the value of sensor is not legal!!")
                
            #set_of_changed_index.add(changed_index)
            features[changed_index] = 1

            
            if len(cells) > 4:
                PersonNumber, WorkString = get_person_and_work(cells[4])
                features[-5] = PersonNumber
                features[-4] = list_of_works.index(WorkString)#WorkNumber
                # i.e. the line is annotated
            features[-3] = cells[0]
            features[-2] = cells[1]
            features[-1] = convert_string_to_datetime(cells[0],cells[1])
            
            if first == True:
                first  = False
            
            if counter < number_of_allowed_samples:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
            
            #reset changed_index to 0
            features[changed_index] = 0
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column

    rows, cols = all_features.shape
    print(rows)
    print(cols)
    
    #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv', 
    np.savetxt(file_address_to_save, np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')


def casas7_to_csv_based_on_each_person_sensor_events_time_Ordered(file_address_to_read, file_address_to_save, number_of_people):
    '''
    0. It is important that the difference between this method and casas7_to_csv_based_on_sensor_events_time_Ordered
       is that in this method the si-on and si-off are considered as two different values of a sensor(events). 
       if they are used, the sensor is set to 1. else 0. 
       In addition, the sensor events of each person is considered as a separate vector.
       each vector represents the set of on and off sensors at that time.
    1. annotated file is processed
    2. just motion, item and door sensors are kept (binary sensors),
         others (i.e. burner, water,temprature and electricity usage) are not. 
    3. 'on' converted to 1
        'off' converted to 0
        'open' converted to 1
        'close' converted to 0
        'present' converted to 0 (because the item is not used)
        'absent' converted to 1
        date and time removed
    4. I supposed the default value of sensors is not important, so set them to 0.
    5. The order of sensor features (51 motion sensors, 
                                     1 item sensor(we have 9 item sensors, but just i03 is used),
                                     9 door sensors (03, 05 , 07, 08, 09, 10 , 12, 14, 15),
                                     Person,
                                     Work,
                                     Date,
                                     Time
    6. Wash_bathtub and Cleaning does not have person number! I supposed the R1 as the person (the kolfat :D)
    7. Because I know the final data has 138039 rows. i defined the array so...
    8. the first instance is name of features
    9. Events are labeled with time and date and the data is sorted based on datatime

    Important: in Tulum 2010 some of the first lines which are not taged for a specific person are removed manually. 
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated","r")
    f = open(file_address_to_read,"r")
    list_of_sensors, number_of_allowed_samples, list_of_works, list_of_person_IDs = get_list_of_allowed_sensors_and_works_in_dataset(file_address_to_read, True)
    
    number_of_columns = len(list_of_sensors) + 5 # 127 for Towr2009
    # 61 features + Person + work + 1 date + 1 time + 1 datetime for ordering data
    features = [0] * number_of_columns 
    each_person_features = [0] * number_of_people
    
    for per in range(number_of_people):
        each_person_features[per] = features
    
    all_features = np.zeros((number_of_allowed_samples, number_of_columns), dtype= object )#np.str)130336 +1
    PersonNumber = -1
    
    counter = -1
    first = True
    for line in f:
        
        cells = line.split()
        
        try:
            feature_column = list_of_sensors.index(cells[2])# get_feature_column(cells[2]) is for Twor2009 dataset and is set manually
        except Exception as e:# i.e. the item is not in list
            feature_column = -1
            
            
        if feature_column != -1:
            counter +=1 
            sensor_value = get_sensor_value(cells[3])
            
            if len(cells) > 4:
                PersonNumber, WorkString = get_person_and_work(cells[4])
                PersonNumber = list_of_person_IDs.index(PersonNumber)
                each_person_features[PersonNumber][-5] = list_of_person_IDs[PersonNumber]
                each_person_features[PersonNumber][-4] = list_of_works.index(WorkString)#WorkNumber
                # i.e. the line is annotated
            each_person_features[PersonNumber][-3] = cells[0]
            each_person_features[PersonNumber][-2] = cells[1]
            each_person_features[PersonNumber][-1] = convert_string_to_datetime(cells[0],cells[1])
            
            each_person_features[PersonNumber][feature_column] = int(sensor_value)
            if first == True:
                first  = False
            
            if counter < number_of_allowed_samples:
                all_features[counter] = each_person_features[PersonNumber]
            else:
                all_features = np.vstack([all_features,each_person_features[PersonNumber]])
            
           
    
    rows, cols = all_features.shape
    print(rows)
    print(cols)
    
    #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv', 
    np.savetxt(file_address_to_save+'_not_sorted.csv', np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    np.savetxt(file_address_to_save, np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')
   



def casas7_activities():
    '''
    1. annotated file is processed
    2. if an activity is ended, set it to 0, 
        if started, set it to 1
    3. I supposed the default value of activities is not important, so set them to 0.
    4. The order of  features: activities, 
                               Person,
                                      
    5. Because I know the final data has  rows. i defined the array so...
    '''
    f = open( join(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"),"r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    # 11 activities + 1 person
    features = [0]* 12
    all_features = np.zeros((1004, 12), dtype= np.int )#np.str)1003 +1
    work_lists = get_work_lists()
    work_lists.append("Person")
    feature_names = work_lists
    
    #used_features = []
    counter = -1
    #print(features)
    first = True
    for line in f:
        
        cells = line.split()
            
        if len(cells) > 4:
            counter+=1
            PersonNumber, WorkActivity = get_person_and_workActivity(cells[4])
            
            features[-1] = PersonNumber
            
            #get index of activity
            WorkIndex = work_lists.index(WorkActivity) 

            if cells[5] == "end":
                features[WorkIndex] = 0
            elif cells[5] == "begin":
                features[WorkIndex] = 1
    
            if first == True:
                first  = False
            
            if counter < 1004:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
  
    
    print(counter) 
    
     
    rows, cols = all_features.shape
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities.csv', 
               all_features, delimiter=',' , fmt='%s')


def casas7_activities_plus_timeLabel():
    '''
    1. annotated file is processed
    2. if an activity is ended, set it to 0, 
        if started, set it to 1
    3. I supposed the default value of activities is not important, so set them to 0.
    4. The order of  features: activities, 
                               Person,
                               Date,
                               Time
                                      
    5. Because I know the final data has  rows. i defined the array so...
    6. the data are not time sorted, because the default data is not, but in the casas7_create_bag_of_activities i sorted them
    '''
    f = open( join(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"),"r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    # 11 activities + 1 person
    features = [0]* 14
    all_features = np.ndarray(shape=(1004, 14), dtype= object)#'|S15')#np.str )#np.str)1003 +1
    
    work_lists = get_work_lists()
    work_lists.append("Person")
    work_lists.append("Date")
    work_lists.append("Time")

    feature_names = work_lists
    
    #used_features = []
    counter = -1
    #print(features)
    first = True
    for line in f:
        
        cells = line.split()
            
        if len(cells) > 4:
            counter+=1
            PersonNumber, WorkActivity = get_person_and_workActivity(cells[4])
            
            features[-3] = PersonNumber
            
            #get index of activity
            WorkIndex = work_lists.index(WorkActivity) 

            if cells[5] == "end":
                features[WorkIndex] = 0
            elif cells[5] == "begin":
                features[WorkIndex] = 1
    
            if first == True:
                first  = False
            
            features[-2] = cells[0] # date
            #print(cells[0])
            #print(features[-2])
            features[-1] = cells[1] # time
            
            if counter < 1004:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
  
    
    print(counter) 
    
        
    rows, cols = all_features.shape
   
    #r1 = dateutil.parser.parse(all_features[1][-1])#time.strptime(all_features[1][-1], '%b %d %Y %I:%M%p')
    #print(r1)
    #print(all_features[1][-1] - all_features[0][-1])#for i in range (rows):
     #   print(all_features[i][-1]) 
   
    print(all_features[0])
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities+time.csv', 
               all_features, delimiter=',' , fmt='%s')



def casas7_create_bag_of_activities(deltaInMinutes , isSave):
    '''
    1. The order of  features: activities, 
                               Person,
                               Date,
                               Time
                                      
    2. each row is constructed by considering the bag of activities. 
       The start time of each activity is the start time of period and number of activities in a period is counted. 
    Parameters:
    deltaInMinutes:
    isSave: Save the returned value to file
    
    '''
    f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities+time.csv","r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    # 11 activities + 1 person
    #features = [0]* 14
    all_features = np.zeros((1004, 15), dtype= object )#np.str)1003 +1
    work_lists = get_work_lists()
    work_lists.append("Person")
    work_lists.append("Date")
    work_lists.append("Time")

    feature_names = work_lists
    
    #used_features = []
    counter = -1
    #print(features)
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
        cells[13] = cells[13].split('\n')[0]# the 0th element is the time, the next is empty
        #print(cells)
        
        converted_cells = [int(i) for i in cells[0:12]]  
        if first == True:
            print(len(converted_cells)) 
        converted_cells.append(cells[12]) # 12 is date
        converted_cells.append(cells[13]) # 13 is time
        #print("counter:{}".format(counter))
        converted_cells.append(convert_string_to_datetime(cells[12], cells[13])) # 14 is datetime object that is used for comparision but is removed before saving file
      
        #print(converted_cells)
        
        #print(type(converted_cells[-1]))
        
        counter+=1
               
        if first == True:
            first  = False
      
        if counter < 1004:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #print(counter) 
     
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities+time_ordered.csv', 
               np.delete(all_features, -1 , 1), delimiter=',' , fmt='%s')# -1 is the number of col and 1 is the number of axis
    
    for each_line in range(counter-1):
        #print("each_line:{}".format(each_line+1))
        for offset in range(each_line-1, -1, -1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                # if personNumbers is equal
                #print("offset:{}".format(offset+1))
                if all_features[each_line][-4] == all_features[offset][-4]:
                    timedelta_in_minutes = timedelta.total_seconds(all_features[each_line][-1] - all_features[offset][-1]) / 60
                    #print("counter:{}, offset:{}, delta:{}".format(counter+1 , offset+1, timedelta_in_minutes))
    
                    #if timedelta_in_minutes < 0:
                     #   print("each_line:{}, offset:{}, delta:{}".format(each_line+1 , offset+1, timedelta_in_minutes))
    
                    # compare delta time in minutes
                    if timedelta_in_minutes < deltaInMinutes:
                        for i in range(11):# 12 is num of features
                            all_features[offset][i] += all_features[each_line][i]
                        
                        #print("each_line: {} , offset: {}".format(each_line+1 , offset+1))
                    else: # ignore to continue 
                        break
                    


    rows, cols = all_features.shape
    
    all_features = np.delete(all_features, -1 , 1)
    
    #print(type(all_features[0]))
    #print(all_features[0])

    #for i in range (1004):
        #print(all_features[i][-2])
    if isSave == True:
        np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_activities_delta_' + str(deltaInMinutes) + 'min.csv', 
             all_features , delimiter=',' , fmt='%s')
    
    return all_features
    


    
def casas7_create_bag_of_sensor_events(deltaInMinutes , isSave):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Date,
                               Time
        activity number is ignored, but exists in input file                                  
    2. each row is constructed by considering the bag of sensor events. 
       The start time of each sensor event is the start time of period and number of sensor events in a period is counted. 
    Parameters:
    deltaInMinutes:
    isSave: Save the returned value to file
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    # 11 activities + 1 person
    #features = [0]* 14
    all_features = np.zeros((130337, 126), dtype= object )#np.str)1003 +1
        
    #used_features = []
    counter = -1
    #print(features)
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
        cells[125] = cells[125].split('\n')[0]# the 0th element is the time, the next is empty
        #print(cells)
        
        converted_cells = [int(i) for i in cells[0:123]]  
        
        if first == True:
            print(len(converted_cells)) 
        
        converted_cells.append(cells[124]) #  is date
        converted_cells.append(cells[125]) #  is time
        #print("counter:{}".format(counter))
        converted_cells.append(convert_string_to_datetime(cells[124], cells[125])) # 126 is datetime object that is used for comparision but is removed before saving file
      
        #print(converted_cells)
        
        #print(type(converted_cells[-1]))
        
        counter+=1
               
        if first == True:
            first  = False
            print(converted_cells[122])
            print(converted_cells[123])
            print(converted_cells[124])
            print(converted_cells[125])
            
      
        if counter < 130337:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #print(counter) 
    #it is sorted in input file 
    #all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    
    #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\activities+time_ordered.csv', 
     #          np.delete(all_features, -1 , 1), delimiter=',' , fmt='%s')# -1 is the number of col and 1 is the number of axis
    
    print("aaaaa")
    for each_line in range(counter-1):
        #print("each_line:{}".format(each_line+1))
    
        for offset in range(each_line-1, -1, -1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                # if personNumbers is equal
                #print("offset:{}".format(offset+1))
                if all_features[each_line][-4] == all_features[offset][-4]:
                    timedelta_in_minutes = timedelta.total_seconds(all_features[each_line][-1] - all_features[offset][-1]) / 60
                    #print("counter:{}, offset:{}, delta:{}".format(counter+1 , offset+1, timedelta_in_minutes))
    
                    #if timedelta_in_minutes < 0:
                     #   print("each_line:{}, offset:{}, delta:{}".format(each_line+1 , offset+1, timedelta_in_minutes))
    
                    # compare delta time in minutes
                    if timedelta_in_minutes < deltaInMinutes:
                        for i in range(121):# 12 is num of features
                            all_features[offset][i] += all_features[each_line][i]
                        
                        #print("each_line: {} , offset: {}".format(each_line+1 , offset+1))
                    else: # ignore to continue 
                        break
                    


    rows, cols = all_features.shape
    
    all_features = np.delete(all_features, -1 , 1)
    
    #print(type(all_features[0]))
    #print(all_features[0])

    #for i in range (1004):
        #print(all_features[i][-2])
    if isSave == True:

        np.savetxt(r'C:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
             all_features , delimiter=',' , fmt='%s')
        
    return all_features
    

def casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes ,number_of_entire_rows, address_to_read, address_for_save, isSave):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Date,
                               Time
        activity number is ignored, but exists in input file                                  
    2. each row is constructed by considering the bag of sensor events, but no overlap exists in deltaT
    
    Parameters:
    ===============
    deltaInMinutes:
    isSave: Save the returned value to file
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    #f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    f = open(address_to_read , "r")
    #print("was opened")
    all_features = np.zeros((number_of_entire_rows, 126), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        #print(counter)
        cells = line.split(',')
        cells[125] = cells[125].split('\n')[0]# the 0th element is the time, the next is empty
        #print(cells)
        
        converted_cells = [int(i) for i in cells[0:123]]  
        
        if first == True:
            print(len(converted_cells)) 
        
        converted_cells.append(cells[124]) #  is date
        converted_cells.append(cells[125]) #  is time
        #print("counter:{}".format(counter))
        converted_cells.append(convert_string_to_datetime(cells[124], cells[125])) # 126 is datetime object that is used for comparision but is removed before saving file
      
        
        counter+=1
               
        if first == True:
            first  = False
            #print(converted_cells[122])
            #print(converted_cells[123])
            #print(converted_cells[124])
            #print(converted_cells[125])
            
      
        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        

    print("start separating person IDs")
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(all_features[: , -4]))
    #print(person_IDs)
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    #print(type(person_data))
    for i in range(number_of_residents):
        person_data[i] = all_features[np.where(all_features[:,-4] == person_IDs[i])]
        #print("*****************\n {}".format(i))
        #print(type(person_data[i]))
        #print(person_data[i].shape)
     
    #save bag of features in deltaT for each person   
    person_bag = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        print("start procesing resident: " , each_person)

        new_counter = 0
        each_line = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person 
        person_bag[each_person] = np.ndarray(shape = (person_data_number_of_rows , 122), dtype = int )
        #print(person_bag[each_person].shape)
        for i in range(122):
            person_bag[each_person][0][i] = person_data[each_person][0][i]
        #print(person_bag[each_person].shape)
        
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            timedelta_in_minutes = timedelta.total_seconds(person_data[each_person][offset][-1] - person_data[each_person][each_line][-1]) / 60
            # compare delta time in minutes
            if timedelta_in_minutes <= deltaInMinutes:
                for i in range(122):# 120 is num of features
                    person_bag[each_person][new_counter][i] += person_data[each_person][offset][i]

            else:  
                each_line = offset
                new_counter += 1
                for i in range(122):
                    person_bag[each_person][new_counter][i] = person_data[each_person][each_line][i]
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_bag[each_person] = np.delete(person_bag[each_person] , range(new_counter + 1 , person_data_number_of_rows + 1 ) , axis = 0)    
         
    #add person_ID column to data
    for person in range(number_of_residents):
        person_col = np.full(( len(person_bag[person]), 1) , person_IDs[person] , dtype = int)  
        person_bag[person] = np.concatenate( (person_bag[person] , person_col) , axis = 1) 
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_bag[0].shape , person_bag[person].shape)
        person_bag[0] = np.concatenate((person_bag[0], person_bag[person]), axis=0)

    
    if isSave == True:
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
        #np.savetxt(r'C:\pgmpy\Bag of sensor events_no overlap_based on different deltas\bag_of_sensor_events_no_overlap_delta_' + str(deltaInMinutes) + 'min.csv', 
        np.savetxt(address_for_save, person_bag[0] , delimiter=',' , fmt='%s' , header = file_header_Twor2009.replace(',Work' , '' ))
    
    
    return person_bag[0]


def casas7_create_bag_of_sensor_events_based_on_activity(number_of_entire_rows, address_to_read, address_for_save, isSave):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Date,
                               Time
    2. each row is constructed by considering the bag of sensor events for each activity
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    #f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    f= open(address_to_read , "r")
    all_features = np.zeros((number_of_entire_rows, 124), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
        
        converted_cells = [int(i) for i in cells[0:124]]  
        
        counter+=1
               
        if first == True:
            first  = False
            print(len(converted_cells))
            print(converted_cells[123])
            
      
        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(all_features[: , -2]))
    #print(person_IDs)
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    #print(type(person_data))
    for i in range(number_of_residents):
        person_data[i] = all_features[np.where(all_features[:,-2] == person_IDs[i])]
        #print("*****************\n {}".format(i))
        #print(type(person_data[i]))
        #print(person_data[i].shape)
     
    #save bag of features in deltaT for each person   
    person_bag = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        each_line = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person 
        person_bag[each_person] = np.ndarray(shape = (person_data_number_of_rows , 124), dtype = int )
        #print(person_bag[each_person].shape)
        for i in range(122):
            person_bag[each_person][0][i] = person_data[each_person][0][i]
        #print(person_bag[each_person].shape)
        last_activity = person_data[each_person][0][-1]
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            # compare delta time in minutes
            if person_data[each_person][offset][-1] == last_activity:
                for i in range(122):# 120 is num of features
                    person_bag[each_person][new_counter][i] += person_data[each_person][offset][i]

            else:  
                each_line = offset
                
                # add activity column value
                person_bag[each_person][new_counter][-1] = last_activity
                #add person column value
                person_bag[each_person][new_counter][-2] = person_IDs[each_person] 
                
                last_activity = person_data[each_person][offset][-1]
                
                new_counter += 1
                for i in range(122):
                    person_bag[each_person][new_counter][i] = person_data[each_person][each_line][i]
                
        #update last row column and activity number
        person_bag[each_person][new_counter][-1] = last_activity
        person_bag[each_person][new_counter][-2] = person_IDs[-1] 
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_bag[each_person] = np.delete(person_bag[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_bag[0].shape , person_bag[person].shape)
        person_bag[0] = np.concatenate((person_bag[0], person_bag[person]), axis=0)

    
    if isSave == True:
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
        #np.savetxt(r'C:\pgmpy\Bag of sensor events_based on activities.csv', 
        np.savetxt(address_for_save, person_bag[0] , delimiter=',' , fmt='%s' , header = file_header)
     
    return person_bag[0]


def casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes, number_of_entire_rows, address_to_read, address_for_save, isSave):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Date,
                               Time
        activity number is ignored, but exists in input file                                  
    2. each row is constructed by considering the bag of sensor events for each activity
     +  a delta for devide an activity
    
    Parameters:
    ===============
    deltaInMinutes: the time delta for dividing the activity
    isSave: Save the returned value to file
    
    '''
    #f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    #f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    f = open(address_to_read, "r")
    all_features = np.zeros((number_of_entire_rows, 127), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
        cells[125] = cells[125].split('\n')[0]# the 0th element is the time, the next is empty
        #print(cells)
        
        converted_cells = [int(i) for i in cells[0:124]]  
        
        #if first == True:
            #print(len(converted_cells)) 
        
        converted_cells.append(cells[124]) #  is date
        converted_cells.append(cells[125]) #  is time
        #print("counter:{}".format(counter))
        converted_cells.append(convert_string_to_datetime(cells[124], cells[125])) # 126 is datetime object that is used for comparision but is removed before saving file
      
        
        counter+=1
               
        if first == True:
            first  = False
            print(converted_cells[121])
            print(converted_cells[122])#person
            print(converted_cells[123])#activity
            print(converted_cells[124])
            print(converted_cells[125])
            print(converted_cells[126])

            
      
        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(all_features[: , -5]))
    #print(person_IDs)
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    #print(type(person_data))
    for i in range(number_of_residents):
        person_data[i] = all_features[np.where(all_features[:,-5] == person_IDs[i])]
        #print("*****************\n {}".format(i))
        #print(type(person_data[i]))
        #print(person_data[i].shape)
     
    #save bag of features in deltaT for each person   
    person_bag = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person 
        person_bag[each_person] = np.ndarray(shape = (person_data_number_of_rows , 124), dtype = int )
        #print(person_bag[each_person].shape)
        for i in range(122):
            person_bag[each_person][0][i] = person_data[each_person][0][i]
        #print(person_bag[each_person].shape)
        
        last_activity = person_data[each_person][0][-4]
        last_calculated_row_for_delta = 0
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            if person_data[each_person][offset][-4] == last_activity:
                timedelta_in_minutes = timedelta.total_seconds(person_data[each_person][offset][-1] - person_data[each_person][last_calculated_row_for_delta][-1]) / 60
                # compare delta time in minutes
                if timedelta_in_minutes <= deltaInMinutes:
                    for i in range(122):# 120 is num of features
                        person_bag[each_person][new_counter][i] += person_data[each_person][offset][i]
                
                else:
                    last_calculated_row_for_delta = offset
                    # add activity column value
                    person_bag[each_person][new_counter][-1] = last_activity
                    #add person column value
                    person_bag[each_person][new_counter][-2] = person_IDs[each_person] 
                    
                    last_activity = person_data[each_person][offset][-4]
                    
                    new_counter += 1
                    for i in range(122):
                        person_bag[each_person][new_counter][i] = person_data[each_person][offset][i]    
                    
            else:                  
                last_calculated_row_for_delta = offset
                # add activity column value
                person_bag[each_person][new_counter][-1] = last_activity
                #add person column value
                person_bag[each_person][new_counter][-2] = person_IDs[each_person] 
                
                last_activity = person_data[each_person][offset][-4]
                
                new_counter += 1
                for i in range(122):
                    person_bag[each_person][new_counter][i] = person_data[each_person][offset][i]
                
        
        #update last row column and activity number
        person_bag[each_person][new_counter][-1] = last_activity
        person_bag[each_person][new_counter][-2] = person_IDs[-1] 
        
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_bag[each_person] = np.delete(person_bag[each_person] , range(new_counter + 1 , person_data_number_of_rows + 1 ) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_bag[0].shape , person_bag[person].shape)
        person_bag[0] = np.concatenate((person_bag[0], person_bag[person]), axis=0)

    
    if isSave == True:
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
        #np.savetxt(r'C:\pgmpy\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_' + str(deltaInMinutes) + 'min.csv', 
        np.savetxt(address_for_save, person_bag[0] , delimiter=',' , fmt='%s' , header = file_header)
     
    return person_bag[0]



def casas7_create_Sequence_of_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes, number_of_entire_rows, address_to_read, address_for_save, isSave, has_header):
    '''
    we imagine in each row of dataset, just one sensor event is active and the dataset is ordered base on time 
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Work,
                               Date,
                               Time
    2. each row is constructed by considering the bag of sensor events for each activity
     +  a delta for devide an activity
     Finally each sample is a sequence per each activity in which  bag of sensor events per each delta are connected to each other
    
    Parameters:
    ===============
    deltaInMinutes: the time delta for dividing the activity
    isSave: Save the returned value to file
    
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False, return_header_separately = False , convert_int_columns_to_int= True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False , return_header_separately = False , convert_int_columns_to_int = True)    
        #header = file_header.split(sep=',')
    
    #add a column for converted date and time (datetime)
    rows , cols = np.shape(data)
    for r in range(rows):
        data[r , -2] = convert_string_to_datetime(data[r , -2], data[r , -1])
    #remove extra columns
    data = np.delete(data , [cols-1] , axis = 1)
  
    
    #seperate each person data in a list 
    person_IDs = list(set(data[: , -3]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-3] == person_IDs[i])]
        
    #save seq of features   
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)
    person_bag = [0] * 122#np.zeros(122 , dtype = np.int)

    for each_person in range(number_of_residents):

        new_counter = 0
        last_calculated_row_for_delta = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person , col2: activity
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 3), dtype = np.ndarray )
        
        print(person_data[each_person][0])
        last_activity = person_data[each_person][0][-2]
        
        for i in range(122):
            person_bag[i] = person_data[each_person][0][i]

        
        #ind = np.where(np.equal(person_data[each_person][0] , 1))
        #print(ind)
        #print(ind[0][0])
        unsaved_sequence = []#header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
         
        #print(unsaved_sequence)
        #person_sequences[each_person][0][0] = [next_element_of_sequence]
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            if person_data[each_person][offset][-2] == last_activity:
                timedelta_in_minutes = timedelta.total_seconds(person_data[each_person][offset][-1] - person_data[each_person][last_calculated_row_for_delta][-1]) / 60
                # compare delta time in minutes
                if timedelta_in_minutes <= deltaInMinutes:
                    for i in range(122):# 120 is num of features
                        person_bag[i] += person_data[each_person][offset][i]
                    
                    
                    if new_counter == 0:
                        print("person_bag:" , person_bag)
                else:
                    last_calculated_row_for_delta = offset
                    
                    unsaved_sequence.append(person_bag.copy())
                    #print(len(unsaved_sequence))
                    
                    for i in range(122):# 120 is num of features
                        person_bag[i] = person_data[each_person][offset][i]
                
            else:
                if new_counter<5:
                    print("offset:" , offset)
                
                unsaved_sequence.append(person_bag.copy())
                # add activity column value
                person_sequences[each_person][new_counter][2] = last_activity
                #add person column value
                person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
                person_sequences[each_person][new_counter][0] = unsaved_sequence
                unsaved_sequence = []
                
                last_activity = person_data[each_person][offset][-2]
                
                last_calculated_row_for_delta = offset
                new_counter += 1
                for i in range(122):# 120 is num of features
                    person_bag[i] = person_data[each_person][offset][i]
                
                
        #update last row column and activity number
        person_sequences[each_person][new_counter][2] = last_activity
        person_sequences[each_person][new_counter][1] = person_IDs[-1] 
        person_sequences[each_person][new_counter][0] = unsaved_sequence
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)

    
    if isSave == True:
        np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person,Activity")
     
    return person_sequences[0]



def get_feature_column(sensor_name):
    '''
    return the column number of feature
    '''
    sensor_type = sensor_name[0]
    if sensor_type in ("M","I","D"):
        pass
    else:
        return -1
    #the second and third characters are the number of sensor
    sensor_number = int(sensor_name[1:3])
    
    if sensor_type == "M":
        #print("M")
        #print(sensor_number)
        return sensor_number -1
    
    elif sensor_type == "I":
        return 51
    
    elif sensor_type == "D":
        if sensor_number == 3 :
            return 52
        if sensor_number == 5 :
            return 53
        if sensor_number == 7 :
            return 54
        if sensor_number == 8 :
            return 55
        if sensor_number == 9 :
            return 56
        if sensor_number == 10 :
            return 57
        if sensor_number == 12 :
            return 58
        if sensor_number == 14 :
            return 59
        if sensor_number == 15 :
            return 60

    else:
    # the sensor type is from ignore lists
        return -1 

def get_sensor_value(sensor_value):
    
    if sensor_value == "ON":
        return '1'
    if sensor_value == "OFF":
        return '0'
    if sensor_value == "OPEN":
        return '1'
    if sensor_value == "CLOSE":
        return '0'
    if sensor_value == "PRESENT":
        return '0'
    if sensor_value == "ABSENT":
        return '1'
    
def get_person_and_work(PersonAndWork):
    
    if PersonAndWork[0]!= "R":
        #the works without person number
        personNumber = '1'
        work = PersonAndWork[0:]
        
    else:
        personNumber = PersonAndWork[1]
        work = PersonAndWork[3:]
     
    '''
    if work == "Cleaning":
        workNumber = '01'
    elif work == "Wash_bathtub":
        workNumber = '02'
    elif work == "work_at_computer":
        workNumber = '03'
    elif work == "sleep":
        workNumber = '04'
    elif work == "bed_to_toilet":
        workNumber = '05'
    elif work == "groom":
        workNumber = '06'
    elif work == "breakfast":
        workNumber = '07'
    elif work == "prepare_dinner":
        workNumber = '08'
    elif work == "prepare_lunch":
        workNumber = '09'
    elif work == "work_at_dining_room_table":
        workNumber = '10'
    elif work == "watch_TV":
        workNumber = '11'
    '''
     
    return (personNumber, work)#workNumber)    
        
def get_person_and_workActivity(PersonAndWork):
    
    if PersonAndWork[0]!= "R":
        #the works without person number
        personNumber = '01'
        work = PersonAndWork[0:]
        
    else:
        personNumber = '0' + PersonAndWork[1]
        work = PersonAndWork[3:]
    
    return (personNumber, work)    
    
    
                
def get_work_lists():
    
    works = ['0']
   # print(len(works))

    f = open( join(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"),"r")
    
    # a regix that says the first character is R and the second is 1 or 2 and the third is _
    regular_exp = r'(R)(1|2)(_)(.*)'
    for line in f:
        #print("hello")
        cells = line.split()
        if len(cells) > 4:
            #print("4")
            find = re.match( regular_exp, cells[4])
            if find:
                w = cells[4][3:]
            else:
                w = cells[4]
            
            if w in works :
                #ind = works.index(w)
                #print(ind)
                pass
            else:
                works.append(w)
                
                
    #print(len(works))
    return works[1:]
    
def convert_string_to_datetime(date_str, time_str):    
    
    #print(date_str , time_str)
    datetime_obj = datetime.datetime.strptime(date_str + time_str,"%Y-%m-%d%H:%M:%S.%f") 
    return datetime_obj

    
def datetime_cherknevis():    
    d1= '2009-02-06'
    t1= '17:15:22.42803'
    d2 = '2009-02-06'
    t2= '17:47:16.229419'
    
    d1_c = datetime.datetime.strptime(d1+t1,"%Y-%m-%d%H:%M:%S.%f") 
    #t1_c = datetime.datetime.strptime(t1,"%H:%M:%S.%f")    
    d2_c = datetime.datetime.strptime(d2+t2,"%Y-%m-%d%H:%M:%S.%f") 
    #t2_c = datetime.datetime.strptime(t2,"%H:%M:%S.%f")    
   
    diffrence = d2_c - d1_c
    m = timedelta.total_seconds(diffrence) / 60
    print(m)

def replace_space_with_comma_in_file():
    f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated","r")
    lines = f.read()
    lines = lines.replace(' ', ',')
    
    with open(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated_morattab.csv", "w") as fw:
        fw.write(lines)
    
    
'''
def load_asia(return_X_y=False):
    """Load and return the asia dataset (classification).


    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')

    with open(join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])

    
   ''' 

def create_sequence_of_sensor_events_based_on_activity(address_to_read,has_header, address_for_save, isSave):
    '''
    we imagine in each row of dataset, just one sensor event is active and the dataset is ordered base on time 
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Work,
                               Date,
                               Time
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = np.int, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = np.int, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=True)    
        header = file_header_Twor2009.split(sep=',')
       
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(data[: , -2]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-2] == person_IDs[i])]
        
    #save bag of features in deltaT for each person   
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person , col2: activity
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 3), dtype = np.ndarray )
        
        last_activity = person_data[each_person][0][-1]
        
        ind = np.where(np.equal(person_data[each_person][0] , 1))
        #print(ind)
        #print(ind[0][0])
        next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
        print(next_element_of_sequence)
        person_sequences[each_person][0][0] = [next_element_of_sequence]
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            if person_data[each_person][offset][-1] == last_activity:
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1
                print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0].append(next_element_of_sequence)

            else:  
                # add activity column value
                person_sequences[each_person][new_counter][2] = last_activity
                #add person column value
                person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
                last_activity = person_data[each_person][offset][-1]
                
                new_counter += 1
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]
                person_sequences[each_person][new_counter][0] = [next_element_of_sequence]
        
                
        #update last row column and activity number
        person_sequences[each_person][new_counter][2] = last_activity
        person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)

    
    if isSave == True:
        np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person,Activity")
     
    return person_sequences[0]


def create_sequence_of_sensor_events_based_on_delta_no_overlap(deltaInMinutes, address_to_read,has_header, address_for_save, isSave):
    '''
    we imagine in each row of dataset, just one sensor event is active and the dataset is ordered base on time 
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Work,
                               Date,
                               Time
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False, return_header_separately = False , convert_int_columns_to_int= True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False , return_header_separately = False , convert_int_columns_to_int = True)    
        header = file_header_Twor2009.split(sep=',')
         
    
    #add a column for converted date and time (datetime)
    rows , cols = np.shape(data)
    for r in range(rows):
        data[r , -2] = convert_string_to_datetime(data[r , -2], data[r , -1])
    #remove date and time columns
    data = np.delete(data , [cols-1] , axis = 1)
     
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(data[: , -3]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-3] == person_IDs[i])]
        
    #save seq of features in deltaT for each person   
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        each_line = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person 
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 2), dtype = np.ndarray )
        ind = np.where(np.equal(person_data[each_person][0] , 1))
        next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
        #print(next_element_of_sequence)
        person_sequences[each_person][0][0] = [next_element_of_sequence]

        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            timedelta_in_minutes = timedelta.total_seconds(person_data[each_person][offset][-1] - person_data[each_person][each_line][-1]) / 60
            # compare delta time in minutes
            if timedelta_in_minutes <= deltaInMinutes:
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
                #print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0].append(next_element_of_sequence)

            else:  
                person_sequences[each_person][new_counter][1] = person_IDs[each_person] 

                each_line = offset
                new_counter += 1
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
                #print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0] = [next_element_of_sequence]
              
        #update last row column and activity number
        person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
        #*********** fek konam person_IDs[each_person] dorost bashe
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_sequences[0]
    for person in range(1, number_of_residents):
        #print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)

    
    if isSave == True:
        np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person")
     
    return person_sequences[0]


def create_sequence_of_sensor_events_based_on_activity_and_delta(deltaInMinutes , address_to_read,has_header, address_for_save, isSave):
    '''
    we imagine in each row of dataset, just one sensor event is active and the dataset is ordered base on time 
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Work,
                               Date,
                               Time
    
    Parameters:
    ===============
    isSave: Save the returned value to file
    
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False, return_header_separately = False , convert_int_columns_to_int= True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False , return_header_separately = False , convert_int_columns_to_int = True)    
        header = file_header_Twor2009.split(sep=',')
    
    #add a column for converted date and time (datetime)
    rows , cols = np.shape(data)
    for r in range(rows):
        data[r , -2] = convert_string_to_datetime(data[r , -2], data[r , -1])
    #remove extra columns
    data = np.delete(data , [cols-1] , axis = 1)
  
    
    #seperate each person data in a list 
    person_IDs = list(set(data[: , -3]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-3] == person_IDs[i])]
        
    #save seq of features   
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):

        new_counter = 0
        last_calculated_row_for_delta = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person , col2: activity
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 3), dtype = np.ndarray )
        
        last_activity = person_data[each_person][0][-2]
        
        ind = np.where(np.equal(person_data[each_person][0] , 1))
        #print(ind)
        #print(ind[0][0])
        next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
        #print(next_element_of_sequence)
        person_sequences[each_person][0][0] = [next_element_of_sequence]
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            if person_data[each_person][offset][-2] == last_activity:
                timedelta_in_minutes = timedelta.total_seconds(person_data[each_person][offset][-1] - person_data[each_person][last_calculated_row_for_delta][-1]) / 60
                # compare delta time in minutes
                if timedelta_in_minutes <= deltaInMinutes:
                    ind = np.where(np.equal(person_data[each_person][offset] , 1))
                    next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1
                    #print(next_element_of_sequence)
                    person_sequences[each_person][new_counter][0].append(next_element_of_sequence)

                else:
                    last_calculated_row_for_delta = offset
                    # add activity column value
                    person_sequences[each_person][new_counter][-1] = last_activity
                    #add person column value
                    person_sequences[each_person][new_counter][-2] = person_IDs[each_person] 
                                        
                    new_counter += 1
                    ind = np.where(np.equal(person_data[each_person][offset] , 1))
                    next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1
                    #print(next_element_of_sequence)
                    person_sequences[each_person][new_counter][0] = [next_element_of_sequence]

                   
                
            else:  
                # add activity column value
                person_sequences[each_person][new_counter][2] = last_activity
                #add person column value
                person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
                last_activity = person_data[each_person][offset][-2]
                
                last_calculated_row_for_delta = offset
                new_counter += 1
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]
                person_sequences[each_person][new_counter][0] = [next_element_of_sequence]
        
                
        #update last row column and activity number
        person_sequences[each_person][new_counter][2] = last_activity
        person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        #print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)

    
    if isSave == True:
        np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person,Activity")
     
    return person_sequences[0]

def create_sequence_of_sensor_events_based_on_number_of_events(number_of_events, address_to_read,has_header, address_for_save, isSave):
    '''
    we imagine in each row of dataset, just one sensor event is active and the dataset is ordered base on time 
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Work,
                               Date,
                               Time
    
    Parameters:
    ===============
    number_of_events:
    address_to_read:
    has_header:
    address_for_save
    isSave: Save the returned value to file
    
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=True, return_header_separately = False , convert_int_columns_to_int= True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = object, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=True , return_header_separately = False , convert_int_columns_to_int = True)    
        header = file_header_Twor2009.split(sep=',')

         
    #add a column for converted date and time (datetime)
    _ , cols = np.shape(data)
    data = np.delete(data , cols - 1 , axis = 1) # remove activity column

     
    #seperate each person data in a list (-1 is the index of person column)
    person_IDs = list(set(data[: , -1]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-1] == person_IDs[i])]
        
    #save seq of features for each person   
    person_sequences = np.zeros(number_of_residents, dtype = np.ndarray)

    for each_person in range(number_of_residents):
        #print("each_line:{}".format(each_line+1))
        new_counter = 0
        each_line = 0
        #initialize 
        person_data_number_of_rows , _ = person_data[each_person].shape
        # create a ndarray with size of all data of the person and each row is an array (for defining sequence) 
        # column0: list(seq) , col1 = person 
        person_sequences[each_person] = np.ndarray(shape = (person_data_number_of_rows , 2), dtype = np.ndarray )
        ind = np.where(np.equal(person_data[each_person][0] , 1))
        next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
        print(next_element_of_sequence)
        person_sequences[each_person][0][0] = [next_element_of_sequence]

        sequence_lengh = number_of_events -1
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
            sequence_lengh -= 1 
            if sequence_lengh >= 0:
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
                print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0].append(next_element_of_sequence)

            else:  
                person_sequences[each_person][new_counter][1] = person_IDs[each_person] 

                each_line = offset
                new_counter += 1
                ind = np.where(np.equal(person_data[each_person][offset] , 1))
                next_element_of_sequence = header[ind[0][0]]# if there is more than one 1, the last is person number. because each row has one 1        
                print(next_element_of_sequence)
                person_sequences[each_person][new_counter][0] = [next_element_of_sequence]
                sequence_lengh = number_of_events - 1
       
                
        #update last row column and activity number
        person_sequences[each_person][new_counter][1] = person_IDs[each_person] 
                
        #remove additional items (because when i created the person_bag[each_person], the size was number of rows )
        person_sequences[each_person] = np.delete(person_sequences[each_person] , range(new_counter + 1 , person_data_number_of_rows) , axis = 0)    
         
    #save all data in person_sequences[0]
    for person in range(1, number_of_residents):
        print(person_sequences[0].shape , person_sequences[person].shape)
        person_sequences[0] = np.concatenate((person_sequences[0], person_sequences[person]), axis=0)

    
    if isSave == True:
        np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person")
     
    return person_sequences[0]



def get_list_of_allowed_sensors_and_works_in_dataset(file_address, return_list_of_persons = False):
    
    '''
    Returns:
    =========
    list of binary sensors (i.e. motion, item and door sensors)
    number of samples which are the changing status of allowed sensors
    list_of_works:
    
    '''
    set_of_sesnors = set()
    set_of_works = set()
    set_of_persons = set()
    f = open( file_address ,"r")
    counter = 0
    for line in f:
        #counter +=1
        cells = line.split()
        #print(cells)
        #try:
        if cells[2][0] in ['M','I','D']:
            set_of_sesnors.add(cells[2])
            counter +=1
            
            if len(cells) > 4:
                if cells[4][0] != 'R':
                    set_of_works.add(cells[4])
                else:
                    set_of_works.add(cells[4][3:])
                    set_of_persons.add(cells[4][1])
                    
            # except Exception as e:
        #    print("Exception in counter: ", counter)
            
    #print(counter)
    list_of_sensors = sorted(list(set_of_sesnors))
    list_of_works = sorted(list(set_of_works))
    set_of_persons = sorted(list(set_of_persons))
    #print(list_of_works)
    if return_list_of_persons:
        return list_of_sensors, counter, list_of_works, set_of_persons

    else:
        return list_of_sensors, counter, list_of_works
    
def create_dataset_each_row_one_feature_on_plus_hour_of_day(file_address = ' ', address_to_save = ' '):
    
    if file_address == ' ':
        file_address = r'E:\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv'
    
    if address_to_save == ' ':
        address_to_save = r'E:\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+hour_of_day.csv'
    
          
    data = read_data_from_CSV_file(dest_file=file_address, data_type = object, 
                                   has_header = False, 
                                   return_as_pandas_data_frame = False, 
                                   remove_date_and_time = False, 
                                   return_header_separately = False, 
                                   convert_int_columns_to_int = True)
    
    rows, cols = np.shape(data)
    data = np.delete(data, obj = cols - 3, axis = 1) # remove the activity col
    data = np.delete(data, obj = (cols -1) - 2, axis = 1) # remove the date col
    
    for r in range(rows):
        data[r,-1] = int(data[r,-1].split(':')[0])
    
    
    np.savetxt(address_to_save, data, delimiter=',' , fmt='%s')


def prepare_each_dataset_and_create_all_bag_and_sequence_of_events():
    pass

def create_sequence_of_events_based_on_different_number_of_events():
   
    address_to_read = r'E:\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv'
    address_for_save = r'E:\pgmpy\Seq of sensor events_based_on_number_of_events\number_of_events={}.csv'
    for events in range(3,10):
        create_sequence_of_sensor_events_based_on_number_of_events(number_of_events = events, 
                                                                   address_to_read = address_to_read, 
                                                                   has_header = False, 
                                                                   address_for_save = address_for_save.format(events), 
                                                                   isSave = True)

def replace_zeros_and_nonzeros_events_by_a_number(address_to_read, address_for_save, isSave, has_header,replaced_value_for_zeros,replaced_value_for_nonzeros):
    '''
    replace the nonzeros events with replace_value
    NOTE: do not replace person and activity columns :) ;)
    Parameters:
    ===============
    deltaInMinutes: the time delta for dividing the activity
    isSave: Save the returned value to file
    
    '''
    data = read_data_from_CSV_file(dest_file = address_to_read, data_type = int, has_header = True, return_as_pandas_data_frame = False , remove_date_and_time = False , return_header_separately = False , convert_int_columns_to_int = True)
    '''
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, data_type = int, has_header = True, return_as_pandas_data_frame = False , remove_date_and_time = False , return_header_separately = False , convert_int_columns_to_int = True)
        #(dest_file = address_to_read, data_type = int, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False, return_header_separately = False , convert_int_columns_to_int= True)    
        has_header = True
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, data_type = int, has_header = has_header, return_as_pandas_data_frame = False , remove_date_and_time=False , return_header_separately = False , convert_int_columns_to_int = True)    
        has_header = False
    '''
    #_, cols = np.shape(data)
    print(data)
    zeros_ind = np.where(np.equal(data[:,0: - 2] , 0))   
    nonzeros_ind = np.where(np.not_equal(data[:,0:-2] , 0))
    
    data[zeros_ind] = replaced_value_for_zeros
    
    data[nonzeros_ind] = replaced_value_for_nonzeros
    
   # print(data)
    #print(np.shape(data))
    if isSave == True:
       np.savetxt(address_for_save, data , delimiter=',' , fmt='%s')
      
            
  

if __name__ == '__main__':
    
    file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
    file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
    file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"
    file_address_Twor2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\twor.2010\data"
    
    file_address_Towr2009_to_save = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"
    file_address_Tulum2010_to_save = r"E:\pgmpy\Tulum2010\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"
    file_address_Tulum2009_to_save = r"E:\pgmpy\Tulum2009\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"
    file_address_Twor2010_to_save = r"E:\pgmpy\Towr2010\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"
    
    #create_dataset_each_row_one_feature_on_plus_hour_of_day()
    #create_sequence_of_events_based_on_different_number_of_events()
    #casas7_to_csv_based_on_sensor_events_time_Ordered(file_address_Twor2010,file_address_Twor2010_to_save)
    #a,_,_  = get_list_of_allowed_sensors_and_works_in_dataset(file_address_Tulum2010)
    #print(len(a))
    #s = "R2_asdf"
    #result = re.match(r'(R)(1|2)(_)(.*)' , s)
    #print(get_work_lists())
    address_to_read = r"E:\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv"

    address_to_save3= r"E:\pgmpy\Seq of sensor events_based on activities\based_on_activities.csv"
    address_to_save1= r"E:\pgmpy\Seq of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    address_to_save2= r"E:\pgmpy\Seq of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    address_to_save4= r"E:\pgmpy\Seq of Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"

    #address_to_save= r"E:\pgmpy\Bag of sensor events_based on activities\based_on_activities.csv"
    address_to_save= r"E:\pgmpy\Bag of sensor events_no overlap_based on different deltas\delta_{delta}min.csv"
    #address_to_save= r"E:\pgmpy\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_{delta}min.csv"
    #casas7_create_bag_of_sensor_events_based_on_activity(number_of_entire_rows= 130337, address_to_read=address_to_read, address_for_save= address_to_save, isSave = True)
    activity_bag = r"E:\pgmpy\Bag of sensor events_based on activities\based_on_activities.csv"
    activity_bag_iccke = r"E:\pgmpy\Bag of sensor events_based on activities\based_on_activities_iccke_approach.csv"
    '''
    replace_zeros_and_nonzeros_events_by_a_number(address_to_read = activity_bag , 
                                                  address_for_save = activity_bag_iccke,
                                                  isSave = True,
                                                  has_header = True,
                                                  replaced_value_for_zeros = -1,
                                                  replaced_value_for_nonzeros = 1)
    '''
    #create_sequence_of_sensor_events_based_on_activity(address_to_read = address_to_read, has_header = False, address_for_save = address_to_save3, isSave = True)

    for i in  range(1,15):#[1600,1800,2000,2500,3000,3500,4000,4500,5000]:#[15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]:
        #range(1100 , 5001 , 100):#
        print(i)
        #casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes=i , number_of_entire_rows= 130337, address_to_read=address_to_read, address_for_save= address_to_save.format(delta = i), isSave = True)
        #casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes=i , number_of_entire_rows= 130337, address_to_read=address_to_read, address_for_save= address_to_save.format(delta = i), isSave = True)
        create_sequence_of_sensor_events_based_on_activity_and_delta(deltaInMinutes = i, address_to_read = address_to_read, has_header = False, address_for_save = address_to_save2.format(delta = i), isSave = True)
        #create_sequence_of_sensor_events_based_on_delta_no_overlap(deltaInMinutes = i, address_to_read = address_to_read, has_header = False, address_for_save = address_to_save1.format(delta = i), isSave = True)
        #casas7_create_Sequence_of_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes = i, number_of_entire_rows = 130337, address_to_read = address_to_read, has_header = False, address_for_save = address_to_save4.format(delta = i), isSave = True)
        
'''
    for i in [15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]:
        casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes=i , isSave= True)
        #address_to_save= r"C:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\train\delta_{}min.csv".format(i)
        #casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes=i , number_of_entire_rows= 117479, address_to_read=address_to_read, address_for_save= address_to_save, isSave = True)
        #print("i: " , i)
        #address_to_save= r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_no overlap_based on different deltas\test\delta_{}min.csv".format(i)
        address_to_save= r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\test\delta_{}min.csv".format(i)

        #casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes= i , number_of_entire_rows= 12858, address_to_read=address_to_read, address_for_save= address_to_save, isSave = True)
        casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes = i, number_of_entire_rows = 12858, address_to_read = address_to_read, address_for_save =address_to_save, isSave = True)
'''
        