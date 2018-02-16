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
from builtins import int
from numpy import dtype
from matplotlib.pyplot import axis
#from nntplib import lines



work_lists = ["0"]

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
    feature_names = ["M01", "M02", "M03", "M04" , "M05" , "M06" , "M07" , "M08" , "M09" , "M10"
                       , "M11", "M12", "M13", "M14" , "M15" , "M16" , "M17" , "M18" , "M19" , "M20"
                       , "M21", "M22", "M23", "M24" , "M25" , "M26" , "M27" , "M28" , "M29" , "M30"
                       , "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40"
                       , "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50"
                       , "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14"
                       , "D15", "PNo", "WNo", "Date" , "Time", "DateTime"]
    
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
    
    #count_of_unordinaries = 0  
    #for i in range(130336):
    #    if timedelta.total_seconds(all_features[i+1][-1]- all_features[i][-1]) < 0: # if time is not ordered
    #        count_of_unordinaries +=1
    #        print(i + 2)
    #print("count_of_unordinaries: {}".format(count_of_unordinaries))
    #all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column

    rows, cols = all_features.shape
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data+time_ordered.csv', 
               np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')

def casas7_to_csv_based_on_sensor_events_time_Ordered():
    '''
    0. It is important that the difference between this method and casas7_to_csv_time_Ordered
       is that in this method the si-on and si-off are considered as two different features. if they 
       are used, the corresponding feature is set to 1. else 0 .
       Ø¯Ø± Ù‡Ø± Ø®Ø· Ù�Ù‚Ø· ÛŒÚ© ÙˆÛŒÚ˜Ú¯ÛŒ ÛŒÚ© Ø§Ø³Øª ØªØ§ Ø¨Ø¹Ø¯Ø§ Ø±ÙˆÛŒ Ø§Ù† Ø¨Ú¯ Ø¨Ø³Ø§Ø²Ù… Ø¨Ø± Ø§Ø³Ø§Ø³ Ø¯Ù„ØªØ§ ØªØ§ÛŒÙ…. 
          ÙˆÙ„ÛŒ Ø¯Ø± Ù‚Ø¨Ù„ÛŒ Ù…Ù…Ú©Ù† Ø§Ø³Øª Ø¯Ø± ÛŒÚ© Ø®Ø· Ø¯Ùˆ ØªØ§ Ø³Ù†Ø³ÙˆØ± ÛŒÚ© Ø¨Ø§Ø´Ù†Ø¯ Ú†ÙˆÙ† 
    ØªØ§ Ø²Ù…Ø§Ù†ÛŒ Ú©Ù‡ ÛŒÚ© Ø³Ù†Ø³ÙˆØ± ØªØºÛŒÛŒØ± Ø­Ø§Ù„Øª Ù†Ø¯Ù‡Ø¯ Ù…Ù‚Ø¯Ø§Ø± Ù‚Ø¨Ù„ÛŒ Ø±Ø§ Ø­Ù�Ø¸ Ù…ÛŒ Ú©Ù†Ø¯. Ø±ÙˆØ´ Ø§ÛŒÙ† Ù…ØªØ¯ Ù…Ø§Ù†Ù†Ø¯ Ø¨Ú¯ Ø¢Ù� Ø§Ú©ØªÛŒÙˆÛŒØªÛŒ Ø§Ø³Øª
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

    
    '''
    f = open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated","r")
    #lines = f.readlin()
    #with open( r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\annotated" , 'w') as converted_file:
    
    features = [0]* 127 # 61 * 2 features + Person + work + 1 date + 1 time + 1 datetime for ordering data
    # for month2, the rows are 64466 (+1)
    all_features = np.zeros((130337, 127), dtype= object )#np.str)130336 +1
    
    #feature_names = ["M01_on", "M01_off", "M02_on", "M02_off", "M03_on", "M03_off", "M04_on" , "M04_off", 
    #                 "M05_on", "M05_off", "M06_on", "M06_off" ,"M07_on", "M07_off" , "M08_on" , "M08_off"
     #                  , "M11", "M12", "M13", "M14" , "M15" , 
      #                 "M16" , "M17" , "M18" , "M19" , "M20"
       #                , "M21", "M22", "M23", "M24" , "M25" , 
        #               "M26" , "M27" , "M28" , "M29" , "M30"
         #              , "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40"
          #             , "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50"
           #            , "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14"
            #           , "D15", "PNo", "WNo", "Date" , "Time", "DateTime"]
    
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
            #set features to 0, because in each time just one feature is 1
            #features = [0]* 127
            #if counter > 1000:
            #    break
            
            sensor_value = get_sensor_value(cells[3])
            
            if sensor_value == 1:
                changed_index = feature_column*2
            else:
                #sensor_value == 0 
                changed_index = feature_column*2 + 1
                
            features[changed_index] = 1

            
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
            
            #reset changed_index to 0
            features[changed_index] = 0
    #count_of_unordinaries = 0  
    #for i in range(130336):
    #    if timedelta.total_seconds(all_features[i+1][-1]- all_features[i][-1]) < 0: # if time is not ordered
    #        count_of_unordinaries +=1
    #        print(i + 2)
    #print("count_of_unordinaries: {}".format(count_of_unordinaries))
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column

    rows, cols = all_features.shape
    print(rows)
    print(cols)
    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv', 
               np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')


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
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
        np.savetxt(r'C:\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
             all_features , delimiter=',' , fmt='%s')
        
    return all_features
    

def casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes , isSave):
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
    f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    
    all_features = np.zeros((130337, 126), dtype= object )#np.str)1003 +1
        
    counter = -1
    
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
      
        
        counter+=1
               
        if first == True:
            first  = False
            #print(converted_cells[122])
            #print(converted_cells[123])
            #print(converted_cells[124])
            #print(converted_cells[125])
            
      
        if counter < 130337:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
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
        np.savetxt(r'C:\pgmpy\Bag of sensor events_no overlap_based on different deltas\bag_of_sensor_events_no_overlap_delta_' + str(deltaInMinutes) + 'min.csv', 
            person_bag[0] , delimiter=',' , fmt='%s')
     
    return person_bag[0]


def casas7_create_bag_of_sensor_events_based_on_activity(isSave):
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
    f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    
    all_features = np.zeros((130337, 124), dtype= object )#np.str)1003 +1
        
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
            
      
        if counter < 130337:
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
        person_bag[each_person] = np.delete(person_bag[each_person] , range(new_counter + 1 , person_data_number_of_rows + 1 ) , axis = 0)    
         
    #save all data in person_bag[0]
    for person in range(1, number_of_residents):
        print(person_bag[0].shape , person_bag[person].shape)
        person_bag[0] = np.concatenate((person_bag[0], person_bag[person]), axis=0)

    
    if isSave == True:
        #np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\bag_of_sensor_events_delta_' + str(deltaInMinutes) + 'min.csv', 
        #    all_features , delimiter=',' , fmt='%s')
        np.savetxt(r'C:\pgmpy\Bag of sensor events_based on activities.csv', 
            person_bag[0] , delimiter=',' , fmt='%s')
     
    return person_bag[0]


def casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes , isSave):
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
    f = open( r"C:\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv","r")
    
    all_features = np.zeros((130337, 127), dtype= object )#np.str)1003 +1
        
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

            
      
        if counter < 130337:
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
        np.savetxt(r'C:\pgmpy\Bag of sensor events_based_on_activity_and_no_overlap_delta\delta_' + str(deltaInMinutes) + 'min.csv', 
            person_bag[0] , delimiter=',' , fmt='%s')
     
    return person_bag[0]




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
        return '01'
    if sensor_value == "OFF":
        return '00'
    if sensor_value == "OPEN":
        return '01'
    if sensor_value == "CLOSE":
        return '00'
    if sensor_value == "PRESENT":
        return '00'
    if sensor_value == "ABSENT":
        return '01'
    
def get_person_and_work(PersonAndWork):
    
    if PersonAndWork[0]!= "R":
        #the works without person number
        personNumber = '01'
        work = PersonAndWork[0:]
        
    else:
        personNumber = '0' + PersonAndWork[1]
        work = PersonAndWork[3:]
        
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
     
    return (personNumber, workNumber)    
        
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

if __name__ == '__main__':
    #casas7_activities_plus_timeLabel()
    #a = [1,2,3,5,4]
    #print(sorted(a))
    #s = "R2_asdf"
    #result = re.match(r'(R)(1|2)(_)(.*)' , s)
    #if result:
     #   print ("result.group() : ", result.group())
      #  print ("result.group(1) : ", result.group(1))
       # print ("result.group(2) : ", result.group(2))
    #replace_space_with_comma_in_file()
    #a = np.array([[1,2,3],[4,5,6],[0,0,1]]) 
    #a1 = a[a[:,-1].argsort()]
    for i in [30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]:
        #casas7_create_bag_of_sensor_events_no_overlap(deltaInMinutes=i , isSave= True)
        casas7_create_bag_of_sensor_events_based_on_activity_and_delta(deltaInMinutes=i , isSave = True)
    #casas7_create_bag_of_sensor_events_based_on_activity(isSave = True)
    #casas7_to_csv_based_on_sensor_events_time_Ordered()