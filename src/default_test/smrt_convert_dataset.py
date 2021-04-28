'''
Created on Sep 1, 2020

@author: Adele
'''
import numpy as np
import json
import os

from dataPreparation import get_list_of_allowed_sensors_and_works_in_dataset
from dataPreparation import convert_string_to_datetime


file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"
file_address_Twor2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\twor.2010\data_edited_by_adele"
file_address_Test3 = r"E:\pgmpy\Test3\annotated"
twor2009_save_dir = r"C:\Users\Adele\Desktop\ref of sMRT\twor.2009"

def create_dataset_json_file(file_address_to_read, file_address_to_save_dir, name, site, vhost = None, exchange = None, topic = None):

    '''
    create the dataset.json metadata file
    '''
    
    list_of_sensors, _, list_of_works, list_of_persons = get_list_of_allowed_sensors_and_works_in_dataset(
                                                            file_address = file_address_to_read,
                                                            return_list_of_persons = True
                                                         )
    
    dataset = {}
    dataset['name'] = name
    
    dataset['activities'] = []
    for activity in list_of_works:
        dataset['activities'].append({
            'name': activity,
            'color': None,
            'is_noise': False,
            'is_ignored': False
        })
    
    dataset['residents'] = []
    for person in list_of_persons:
        dataset['residents'].append({
            'name': 'R' + person,
            'color': None
        })
    
    dataset['site'] = site
    dataset['vhost'] = vhost
    dataset['exchange'] = exchange
    dataset['topic'] = topic
        
    file_address_to_save = os.path.join(file_address_to_save_dir, 'data', 'dataset.json')
    with open(file_address_to_save, 'w') as outfile:
        outfile.write(json.dumps(dataset, indent = 2))


def create_site_json_file(file_address_to_read, file_address_to_save_dir, floorplan, name, timezone = 'America/Los_Angeles'):

    '''
    create the site.json metadata file
    '''
    
    list_of_sensors, _, _ = get_list_of_allowed_sensors_and_works_in_dataset(file_address = file_address_to_read)
    
    site = {}
    site['floorplan'] = floorplan
    site['name'] = name
    
    site['sensors'] = []
    for sensor in list_of_sensors:
        site['sensors'].append({
            'name': sensor,
            'types': [get_sensor_type(sensor)],
            'locX': 0.0,
            'locY': 0.0,
            'sizeX': None,
            'sizeY': None,
            'description': None,
            'serial': None,
            'tag': None
            })
    
    site['timezone'] = timezone
        
    file_address_to_save = os.path.join(file_address_to_save_dir,'site' ,'site.json')
    with open(file_address_to_save, 'w') as outfile:
        outfile.write(json.dumps(site, indent = 2))

    
def get_sensor_type(sensor):

    if sensor[0] == 'M':
        return 'MotionSensor'
    if sensor[0] == 'I':
        return 'ItemSensor'
    if sensor[0] == 'D':
        return 'DoorSensor'
    else:
        return 'OtherType'
       
       
def convert_casas_dataset_to_pycasas_events_file(file_address_to_read, file_address_to_save_dir):
    
    """
    """
    
    f = open(file_address_to_read,"r")
    list_of_sensors, number_of_allowed_samples, list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(file_address_to_read)
    #print(list_of_sensors)
    number_of_columns = 7# 6+1 (1 for datetime object used to sort data based on time)
    
    features = [0] * number_of_columns 
    all_features = np.zeros((number_of_allowed_samples, number_of_columns), dtype= object )
    
    counter = -1
    first = True
    active_activities = []
    active_residents = []
    
    for line in f:
        
        cells = line.split()
        
        try:
            feature_column = list_of_sensors.index(cells[2])
        except Exception as e:# i.e. the item is not in list
            feature_column = -1
            
        if len(cells) > 4:
                
                if cells[5] == 'begin':
                    if cells[4][0] == 'R':
                        person = cells[4].split('_')[0]
                        activity = cells[4][len(person)+1:] # remain of the string is the activity
                        active_activities.append(activity)
                        active_residents.append(person)
                    else:
                        active_activities.append(cells[4])
                        
        if feature_column != -1:
            counter +=1
            
            date_details = cells[0].split('-')
            date = date_details[1] + '/' + date_details[2] + '/' + date_details[0] # month/day/year
            features[0] = date + " " + cells[1] + " " + "-08:00" # date time
            features[1] = cells[2] # sensor
            features[2] = cells[3] # message
            features[3] = ';'.join(sorted(list(set(active_residents))))#residents
            features[4] = ';'.join(sorted(list(set(active_activities))))#activities
            features[5] = get_sensor_type(cells[2])#sensor_type
            features[6] = convert_string_to_datetime(cells[0],cells[1])

        if len(cells) > 4 and cells[5] == 'end':
            if cells[4][0] == 'R':
                person = cells[4].split('_')[0]
                activity = cells[4][len(person)+1:] # remain of the string is the activity
                active_activities.remove(activity)
                active_residents.remove(person)
            else:
                active_activities.remove(cells[4])

        if feature_column != -1:
            
            if first == True:
                first  = False
            
            if counter < number_of_allowed_samples:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
            
    file_address_to_save = os.path.join(file_address_to_save_dir, 'data', 'events.csv')
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column
    np.savetxt(file_address_to_save, np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')


def convert_old_casas_datasets_to_new_format():

    convert_casas_dataset_to_pycasas_events_file(file_address_Towr2009, twor2009_save_dir)
 
    create_dataset_json_file(file_address_to_read = file_address_Towr2009, 
                             file_address_to_save_dir = twor2009_save_dir, 
                             name = 'twor.2009', 
                             site = 'twor.2009')
   
    create_site_json_file(file_address_to_read = file_address_Towr2009, 
                          file_address_to_save_dir = twor2009_save_dir, 
                          floorplan = 'TWOR_2009.png', 
                          name = 'twor.2009')
                          
                          
if __name__ == "__main__":
    
    convert_old_casas_datasets_to_new_format()