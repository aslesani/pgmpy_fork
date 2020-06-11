# -*- coding: utf-8 -*-
"""
Created on Thursday June 11 2020

@author: Adele
"""
import numpy  as np
from datetime import timedelta
import datetime
from dataPreparation import get_list_of_allowed_sensors_and_works_in_dataset
from dataPreparation import file_header_Twor2009, file_header_Tulum2009, file_header_Tulum2010
import pandas as pd 
from read_write import read_data_from_CSV_file
import re


file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"

def get_list_of_involved_activities_for_each_sensor(dataset_address, dataset_name):
    
    #list_of_sensors, _ , list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(file_address = dataset_address, return_list_of_persons = False)
    #create an empty list for each sensor
    
    sensors_and_activities = {}# empty dict  
    
    address_of_sequences_of_activities = r"E:\pgmpy\{}\Seq of sensor events_based on activities\based_on_activities.csv".format(dataset_name)
    with open(address_of_sequences_of_activities,'r') as seq_of_activities:
        is_header_line = True
        for line in seq_of_activities:
            
            #skip the header
            if is_header_line:
                is_header_line = False
                continue
                
            line = re.split('\[|\]|,| |\n', line)
            
            line = list(filter(lambda a: a != '', line))#remove empty strings from line
            workNumber = line[-1]
            line = [i.replace('\'','') for i in line[0:len(line)-2]]# remove the two last items because they are personID and WorkID
            for i in range(len(line)):
                sensor_name = re.split('_on|_off', line[i])[0]
                if sensor_name in sensors_and_activities:
                    last_set_of_activities = sensors_and_activities[sensor_name]
                    last_set_of_activities.add(workNumber)
                    sensors_and_activities[sensor_name] = last_set_of_activities
                else:
                    sensors_and_activities[sensor_name] = {workNumber}

    
    for sensor_name in sensors_and_activities: 
        print(sensor_name, sorted(list(sensors_and_activities[sensor_name])))


if __name__ == "__main__":

    get_list_of_involved_activities_for_each_sensor(file_address_Towr2009, "Twor2009")
    
    #get_list_of_involved_activities_for_each_sensor(file_address_Tulum2010, "Tulum2010")
    
