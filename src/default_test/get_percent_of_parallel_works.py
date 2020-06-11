# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:43:20 2019

@author: Adele
"""
from dataPreparation import convert_string_to_datetime
import numpy  as np
from datetime import timedelta
import datetime

def get_list_of_durations(number_of_entire_rows, dataset_name):
    '''
    1. The order of  features: sensor events (for each sensor on and off), 
                               Person,
                               Date,
                               Time
    
    
    This function works for every multi-rsident dataset
	'''
	
    address_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
    f= open(address_to_read , "r")
    all_features = None#np.zeros((number_of_entire_rows, 124), dtype= object )#np.str)1003 +1
        
    counter = -1
    
    first = True
    for line in f:
        #print(line)
        cells = line.split(',')
       
        if first:
            number_of_columns = 3 # Person + Activity + datetime
            all_features = np.zeros((number_of_entire_rows, number_of_columns), dtype= object )#np.str)1003 +1


        converted_cells = []
        converted_cells.append(int(cells[-4]))#person
        converted_cells.append(int(cells[-3]))#activity
        #print(cells[-2], cells[-1])
        converted_cells.append(convert_string_to_datetime(cells[-2], cells[-1].split('\n')[0])) #datetime
        counter+=1
               
        if first == True:
            first  = False
           
      
        if counter < number_of_entire_rows:
            all_features[counter] = converted_cells
        else:
            all_features = np.vstack([all_features,converted_cells])
        
    #seperate each person data in a list (-4 is the index of person column)
    person_IDs = list(set(all_features[: , -3]))
    #print(person_IDs)
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    durations = np.zeros(number_of_residents, dtype = np.ndarray)
    #print(type(person_data))
    for i in range(number_of_residents):
        person_data[i] = all_features[np.where(all_features[:,-3] == person_IDs[i])]
       
     
    #separate each person activities based on start and end time of items
    for per in range(number_of_residents):
        start_time = person_data[per][0][-1]
        activity = person_data[per][0][-2]
        durations[per] = []
        for i in range(1,len(person_data[per])):
            if person_data[per][i][-2] != activity:
                durations[per].append({"start_time": start_time, "end_time": person_data[per][i-1][-1]})
                start_time = person_data[per][i][-1]
                activity = person_data[per][i][-2]
	
    return durations


def return_common_time_in_seconds(duration1, duration2):
    '''
	return the duration of common time
	'''
    max_start_time = duration1["start_time"]
    if duration2["start_time"] > max_start_time:
	    max_start_time = duration2["start_time"]
		
    min_end_time = duration1["end_time"]
    if duration2["end_time"] < min_end_time:
	    min_end_time = duration2["end_time"]

    if max_start_time < min_end_time:
	    return timedelta.total_seconds(min_end_time - max_start_time)
		
    else:
	    return 0

def test_return_common_time_duration():

	dt1 =  {'start_time': datetime.datetime(2009, 4, 3, 17, 50, 00, 0), 'end_time': datetime.datetime(2009, 4, 3, 17, 52, 0, 0)}
	dt2 =  {'start_time': datetime.datetime(2009, 4, 3, 17, 51, 00, 0), 'end_time': datetime.datetime(2009, 4, 3, 17, 59, 0, 0)}
	
	common_time = return_common_time_in_seconds(dt1,dt2)
	print(common_time)

	
def get_duration_of_parallel_works(duratinos):
    '''
    this function work for 2-resident dataset
	
	len(duratinos) is number of residents
	'''
    
    p1_index = 0
    p2_index = 0
	
    p1_number_of_activites = len(duratinos[0])
    p2_number_of_activites = len(duratinos[1])

    whole_common_time = 0
    while (p1_index != (p1_number_of_activites-1) and p2_index != (p2_number_of_activites-1)):
        common_time = return_common_time_in_seconds(duratinos[0][p1_index] , duratinos[1][p2_index])
        whole_common_time = whole_common_time + common_time
		
        if duratinos[0][p1_index]["end_time"] <= duratinos[1][p2_index]["end_time"]:
            p1_index += 1
        else:
            p2_index += 1
	
    print(whole_common_time)
    return whole_common_time

def calculate_the_whole_duration_of_every_person_activities(durations):
    
    sum_of_times = 0
    for per in range(len(durations)):
        for i in range(len(durations[per])):
            sum_of_times += timedelta.total_seconds(durations[per][i]["end_time"] - durations[per][i]["start_time"] )
	
    return sum_of_times	
		

def get_percent_of_parallel_sensor_events(number_of_entire_rows, dataset_name, whole_common_time):
    '''
	count number of events in common time of performed activites
	'''
	# بعدا اصلاح شود. داده ای که به آن پاس داده میشود اشتباه است.
    address_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
    f= open(address_to_read , "r")
    counter = 0
    print(whole_common_time)
    durations_index = 0
	
    for line in f:    
        cells = line.split(',')
       
        date_time = convert_string_to_datetime(cells[-2], cells[-1].split('\n')[0]) #datetime
        if date_time < whole_common_time[durations_index]["start_time"]:
            continue
		
        if date_time <= whole_common_time[durations_index]["end_time"]:
            counter += 1
		
        if date_time > whole_common_time[durations_index]["end_time"]:
            durations_index += 1
            if date_time >= whole_common_time[durations_index]["start_time"] and date_time <= whole_common_time[durations_index]["end_time"]:
                counter += 1
       
    return counter/number_of_entire_rows
	
def get_percent_of_parallel_works(number_of_entire_rows, dataset_name):

    durations = get_list_of_durations(number_of_entire_rows, dataset_name)
    
    whole_common_time = get_duration_of_parallel_works(durations)
    whole_duration_of_each_person = calculate_the_whole_duration_of_every_person_activities(durations)
	
    #percent = whole_common_time/whole_duration_of_each_person
    #print("percent of parallel works:", percent)
	
    percent_of_parallel_sensor_events = get_percent_of_parallel_sensor_events(number_of_entire_rows, dataset_name, whole_common_time)
    print("percent_of_parallel_sensor_events:", percent_of_parallel_sensor_events)
	
    return percent
	
	
    

if __name__ == "__main__":
    
   
    #list_of_Test_sensors, number_of_allowed_samples, list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(Test)
    #print(list_of_Test_sensors)
    #a = create_header_string(list_of_Test_sensors)
    dataset_name, rows = "Twor2009" , 130337
    #dataset_name, rows = "Tulum2009", 277157
    #dataset_name, rows = "Tulum2010", 1014507
    print(dataset_name)
    get_percent_of_parallel_works(rows, dataset_name)
    #test_return_common_time_duration()