# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 2019

@author: Adele
"""
import numpy  as np
from datetime import timedelta
import datetime
from dataPreparation import get_list_of_allowed_sensors_and_works_in_dataset
from dataPreparation import file_header_Twor2009, file_header_Tulum2009, file_header_Tulum2010
import pandas as pd 
from read_write import read_data_from_CSV_file


file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"

#file_header_Twor2009 = "D03_on,D03_off,D05_on,D05_off,D07_on,D07_off,D08_on,D08_off,D09_on,D09_off,D10_on,D10_off,D12_on,D12_off,D14_on,D14_off,D15_on,D15_off,I03_on,I03_off,M01_on,M01_off,M02_on,M02_off,M03_on,M03_off,M04_on,M04_off,M05_on,M05_off,M06_on,M06_off,M07_on,M07_off,M08_on,M08_off,M09_on,M09_off,M10_on,M10_off,M11_on,M11_off,M12_on,M12_off,M13_on,M13_off,M14_on,M14_off,M15_on,M15_off,M16_on,M16_off,M17_on,M17_off,M18_on,M18_off,M19_on,M19_off,M20_on,M20_off,M21_on,M21_off,M22_on,M22_off,M23_on,M23_off,M24_on,M24_off,M25_on,M25_off,M26_on,M26_off,M27_on,M27_off,M28_on,M28_off,M29_on,M29_off,M30_on,M30_off,M31_on,M31_off,M32_on,M32_off,M33_on,M33_off,M34_on,M34_off,M35_on,M35_off,M36_on,M36_off,M37_on,M37_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M42_on,M42_off,M43_on,M43_off,M44_on,M44_off,M45_on,M45_off,M46_on,M46_off,M47_on,M47_off,M48_on,M48_off,M49_on,M49_off,M50_on,M50_off,M51_on,M51_off,Person,Work"
#file_header_Twor2010 = "D001_on,D001_off,D002_on,D002_off,D003_on,D003_off,D004_on,D004_off,D005_on,D005_off,D006_on,D006_off,D007_on,D007_off,D008_on,D008_off,D009_on,D009_off,D010_on,D010_off,D011_on,D011_off,D012_on,D012_off,D013_on,D013_off,D014_on,D014_off,D015_on,D015_off,I006_on,I006_off,I010_on,I010_off,I011_on,I011_off,I012_on,I012_off,M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,M032_on,M032_off,M033_on,M033_off,M034_on,M034_off,M035_on,M035_off,M036_on,M036_off,M037_on,M037_off,M038_on,M038_off,M039_on,M039_off,M040_on,M040_off,M041_on,M041_off,M042_on,M042_off,M043_on,M043_off,M044_on,M044_off,M045_on,M045_off,M046_on,M046_off,M047_on,M047_off,M048_on,M048_off,M049_on,M049_off,M050_on,M050_off,M051_on,M051_off,Person,Work"
#file_header_Tulum2009 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,Person,Work"
#file_header_Tulum2010 = "M001_on,M001_off,M002_on,M002_off,M003_on,M003_off,M004_on,M004_off,M005_on,M005_off,M006_on,M006_off,M007_on,M007_off,M008_on,M008_off,M009_on,M009_off,M010_on,M010_off,M011_on,M011_off,M012_on,M012_off,M013_on,M013_off,M014_on,M014_off,M015_on,M015_off,M016_on,M016_off,M017_on,M017_off,M018_on,M018_off,M019_on,M019_off,M020_on,M020_off,M021_on,M021_off,M022_on,M022_off,M023_on,M023_off,M024_on,M024_off,M025_on,M025_off,M026_on,M026_off,M027_on,M027_off,M028_on,M028_off,M029_on,M029_off,M030_on,M030_off,M031_on,M031_off,Person,Work"

def check_the_neighberhood_of_two_sensors(maps, sensor1, sensor2):
    '''
    Parameters:
    ============
    maps: pandas dataframe, the maps of sensors created by file of sensor events
    sensor1: the sensor name
    sensor2: the sensor name
    
    return:
    ========
    bool: True, if the sensors are neighboured, else false
    '''
    if maps.at[sensor1, sensor2] == 1 or maps.at[sensor2, sensor1] == 1:
        return True
    else:
        return False
    
    
def create_map_of_sensors(dataset_name, has_header, address_for_save, isSave, header):
    ''' 
    consider just on events. becuase maybe the off events are logged with delay.
    In addition, each person should be considered separately, to get the map of sensors.	
	'''
	
    address_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
	
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = address_to_read, 
		                                        data_type = np.int, 
												has_header = has_header, 
												return_as_pandas_data_frame = False , 
												remove_date_and_time=True)    
    else:
        data = read_data_from_CSV_file(dest_file = address_to_read, 
		                               data_type = np.int, 
									   has_header = has_header, 
									   return_as_pandas_data_frame = False, 
									   remove_date_and_time=True)    
        header = header.split(sep=',')
       
    person_IDs = list(set(data[: , -2]))
    number_of_residents = len(person_IDs)
    person_data = np.zeros(number_of_residents, dtype = np.ndarray)
    for i in range(number_of_residents):
        person_data[i] = data[np.where(data[:,-2] == person_IDs[i])]

    maps = pd.DataFrame()#for saving the maps of sensors   
    
    for each_person in range(number_of_residents):
        ind = np.where(np.equal(person_data[each_person][0] , 1))
        #print(ind[0])
        last_sensor = header[ind[0][0]].split('_')[0]# if there is more than one 1, the last is person number. because each row has one 1. 
		                                             # in addition, it ignores the on or off status of sensor     
        print(last_sensor)
        
        for offset in range(1, len(person_data[each_person]), 1): # range(start, end, step) sharte end ine k bozorgtar bashe pas bayad yeki az akhari kamtar begiram
                
            ind = np.where(np.equal(person_data[each_person][offset] , 1))
            sensor_event = header[ind[0][0]]
            if sensor_event.split('_')[1] == 'on':
                current_sensor = sensor_event.split('_')[0]
                maps.at[last_sensor, current_sensor] = 1
                last_sensor = current_sensor
				
    maps = maps.fillna(0).astype(int)#replace NaN values with 0 and convert all columns from float! to int 
    
    #pd.set_option('display.max_columns', 100) # print the entire dataframe, not the truncated one :)
    #pd.set_option('display.max_colwidth', 30)
    #print(maps)
    
    maps.sort_index(axis = 0, inplace = True)#sort based on labels of rows
    maps.sort_index(axis = 1, inplace = True)#sort based on labels of columns
    #print("after sort:")
    #print(maps)
    #pd.reset_option('display.max_columns')# reset the display.max_rows, because it changes the options of pnadas dataframe not the maps dataframe
    #pd.reset_option('display.max_colwidth')
    
    if isSave:
        maps.to_csv(path_or_buf = address_for_save)
    #if isSave == True:
    #    np.savetxt(address_for_save, person_sequences[0] , delimiter=',' , fmt='%s' , header = "Sequence,Person,Activity")
     

def read_pandas_dataframe_from_csv(address):
    
    df = pd.read_csv(address, header = 0, index_col = 0)# header indicated the line number of header file(or columns names) and index_col indicates the column number of row names (row indexes)
    return df
    

def test_check_the_neighberhood_of_two_sensors(maps_address, sensor1, sensor2):
    
    maps = read_pandas_dataframe_from_csv(maps_address)
    result = check_the_neighberhood_of_two_sensors(maps, sensor1, sensor2)
    print(result)

def convert_each_row_one_features_file_to_event_names(dataset_name):
    '''
    convert sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv to a file that 
    indicates the sensor event explicitly, not a feature vector with one 1 value :D
    '''
    address_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
    address_for_save = r"E:\pgmpy\{}\sensor_data_event_names_time_ordered.csv".format(dataset_name)

def ckeck_initial_occlusion(maps_of_sensors, sensor_events_file_to_read, has_header, header):
    '''
    this function ckecks wherther there is initial pcclusion or not. 
    initial occlusion occures when a new person starts producing an event 
    but the event is for a sensor in neighbourhood of other persons current fired sensors.
    
    Return:
    ========
    bool: True if there is initial occlusion, else False
    '''
        
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = sensor_events_file_to_read, 
		                                        data_type = np.int, 
												has_header = has_header, 
												return_as_pandas_data_frame = False , 
												remove_date_and_time=True)    
    else:
        data = read_data_from_CSV_file(dest_file = sensor_events_file_to_read, 
		                               data_type = np.int, 
									   has_header = has_header, 
									   return_as_pandas_data_frame = False, 
									   remove_date_and_time=True)    
        header = header.split(sep=',')
       
    number_of_residents = len(list(set(data[: , -2])))
    number_of_seen_residents = 0
    
    last_fired_sensor_of_each_person = dict()
    last_seen_person_ID = -1
    
    for i in range(len(data)):
        current_person_ID = data[i,-2]
        ind = np.where(np.equal(data[i] , 1))
        current_sensor_event = header[ind[0][0]]
        
        if current_sensor_event.split('_')[1] == 'on':# we just consider on events
            current_sensor = current_sensor_event.split('_')[0]
            
            if current_person_ID != last_seen_person_ID:
                if current_person_ID not in last_fired_sensor_of_each_person: # if the person is not seen before
                    number_of_seen_residents += 1
                    #check if there is occlusion between the new person and the previous ones
                    for PID, last_person_sensor in last_fired_sensor_of_each_person.items():
                        if current_sensor == last_person_sensor or check_the_neighberhood_of_two_sensors(maps_of_sensors, current_sensor, last_person_sensor):
                            print(r"occlusion of Person {current_person_ID} and {PID}".format(current_person_ID = current_person_ID, PID = PID))
                            return True
                    
                    if number_of_seen_residents == number_of_residents: # it means that there is no INITIAL occlusion between persons
                        print(last_fired_sensor_of_each_person)
                        return False
                    
                last_fired_sensor_of_each_person[current_person_ID] = current_sensor # if the person exists, updates the last fired sensor, else creates it
                last_seen_person_ID = current_person_ID 
            
            else: # if she/he is the previous one
                last_fired_sensor_of_each_person[current_person_ID] = current_sensor 
        
    return False
    
    
def test_ckeck_initial_occlusion(dataset_name):
    
    sensor_map_file = r"E:\pgmpy\{}\sensor_maps.csv".format(dataset_name)
    maps_of_sensors = read_pandas_dataframe_from_csv(sensor_map_file)
    
    sensor_events_file_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
	    
    result = ckeck_initial_occlusion(maps_of_sensors, sensor_events_file_to_read, has_header = False, header = eval("file_header_" + dataset_name))
    print(result)


def percent_of_occlusion(maps_of_sensors, sensor_events_file_to_read, has_header, header):
    
    '''
    there is occlusion when the fired sensor is in neighbourhood of more than one person's previous sensor. 
    or if the new arrival person is in neighbourhood of one of previuos persons's last fired sensor. 
    the second one is ignored at this code. I just check it to not exist. 
    if I detect an occlusion, I sohuld reset the last sensor of the person involved in occlusion to continue proccessing other events and calculate the percent of ocllusion. 
    Note: I just consider on events
    '''
    
    number_of_occlusions = 0
    
    is_init_occ = ckeck_initial_occlusion(maps_of_sensors, sensor_events_file_to_read, has_header, header)
    
    if is_init_occ:
        raise Exception("There is initial occlusion. We can not process data in this version of code.")
    
    if has_header:
        header , data = read_data_from_CSV_file(dest_file = sensor_events_file_to_read, 
		                                        data_type = np.int, 
												has_header = has_header, 
												return_as_pandas_data_frame = False , 
												remove_date_and_time=True)    
    else:
        data = read_data_from_CSV_file(dest_file = sensor_events_file_to_read, 
		                               data_type = np.int, 
									   has_header = has_header, 
									   return_as_pandas_data_frame = False, 
									   remove_date_and_time=True)    
        header = header.split(sep=',')
       
    
    last_fired_sensor_of_each_person = []
    occlusion_line_numbers = []
    #find the first fired sensor
    for i in range(len(data)):
        ind = np.where(np.equal(data[i] , 1))
        current_sensor_event = header[ind[0][0]]
        
        if current_sensor_event.split('_')[1] == 'on':# we just consider on events
            current_sensor = current_sensor_event.split('_')[0]
            last_fired_sensor_of_each_person.append(current_sensor)
            first_on_event_index = i
            break
        
    for i in range(first_on_event_index+1, len(data)):
        ind = np.where(np.equal(data[i] , 1))
        current_sensor_event = header[ind[0][0]]
        
        if current_sensor_event.split('_')[1] == 'on':# we just consider on events
            current_sensor = current_sensor_event.split('_')[0]
            number_of_local_neighbours = 0
            neighbour_senosrs = []
            
            #check if there is occlusion between the new sensor event and the previous ones
            for s in last_fired_sensor_of_each_person:
                if current_sensor == s or check_the_neighberhood_of_two_sensors(maps_of_sensors, current_sensor, s):
                    number_of_local_neighbours += 1
                    neighbour_senosrs.append(s)
                    
            if number_of_local_neighbours == 0:
                # it means that the person is new
                last_fired_sensor_of_each_person.append(current_sensor)
            elif number_of_local_neighbours == 1:
                # that means that the sensor is the next sensor in sequence
                last_fired_sensor_of_each_person[last_fired_sensor_of_each_person.index(neighbour_senosrs[0])] = current_sensor
            elif number_of_local_neighbours > 1:
                #that means there is an occlusion unfortunately :D
                occlusion_line_numbers.append(i)
                number_of_occlusions += 1
                
    return number_of_occlusions, len(data), occlusion_line_numbers           
        
def test_percent_of_occlusion(dataset_name):
    
    sensor_map_file = r"E:\pgmpy\{}\sensor_maps.csv".format(dataset_name)
    maps_of_sensors = read_pandas_dataframe_from_csv(sensor_map_file)
    
    sensor_events_file_to_read = r"E:\pgmpy\{}\sensor_data_each_row_one_features_is_one_on_and_off+time_ordered.csv".format(dataset_name)
	    
    try:
        number_of_occlusions, number_of_events, occlusion_line_numbers = percent_of_occlusion(maps_of_sensors, 
                                                                                              sensor_events_file_to_read, 
                                                                                              has_header = False, 
                                                                                              header = eval("file_header_" + dataset_name))
        print("number_of_occlusions:" , number_of_occlusions, "number_of_events:", number_of_events)
        #print("occlusion_line_numbers:\n", occlusion_line_numbers)
    except Exception as e:
        print(e)

if __name__ == "__main__":

    #Twor2009_sorted_header = "D03_on,D03_off,D05_on,D05_off,D07_on,D07_off,D08_on,D08_off,D09_on,D09_off,D10_on,D10_off,D12_on,D12_off,D14_on,D14_off,D15_on,D15_off,I03_on,I03_off,M01_on,M01_off,M02_on,M02_off,M03_on,M03_off,M04_on,M04_off,M05_on,M05_off,M06_on,M06_off,M07_on,M07_off,M08_on,M08_off,M09_on,M09_off,M10_on,M10_off,M11_on,M11_off,M12_on,M12_off,M13_on,M13_off,M14_on,M14_off,M15_on,M15_off,M16_on,M16_off,M17_on,M17_off,M18_on,M18_off,M19_on,M19_off,M20_on,M20_off,M21_on,M21_off,M22_on,M22_off,M23_on,M23_off,M24_on,M24_off,M25_on,M25_off,M26_on,M26_off,M27_on,M27_off,M28_on,M28_off,M29_on,M29_off,M30_on,M30_off,M31_on,M31_off,M32_on,M32_off,M33_on,M33_off,M34_on,M34_off,M35_on,M35_off,M36_on,M36_off,M37_on,M37_off,M38_on,M38_off,M39_on,M39_off,M40_on,M40_off,M41_on,M41_off,M42_on,M42_off,M43_on,M43_off,M44_on,M44_off,M45_on,M45_off,M46_on,M46_off,M47_on,M47_off,M48_on,M48_off,M49_on,M49_off,M50_on,M50_off,M51_on,M51_off,Person,Work"
    #'D03_off, D03_on, D05_off, D05_on, D07_off, D07_on, D08_off, D08_on, D09_off, D09_on, D10_off, D10_on, D12_off, D12_on, D14_off, D14_on, D15_off, D15_on, I03_off, I03_on, M01_off, M01_on, M02_off, M02_on, M03_off, M03_on, M04_off, M04_on, M05_off, M05_on, M06_off, M06_on, M07_off, M07_on, M08_off, M08_on, M09_off, M09_on, M10_off, M10_on, M11_off, M11_on, M12_off, M12_on, M13_off, M13_on, M14_off, M14_on, M15_off, M15_on, M16_off, M16_on, M17_off, M17_on, M18_off, M18_on, M19_off, M19_on, M20_off, M20_on, M21_off, M21_on, M22_off, M22_on, M23_off, M23_on, M24_off, M24_on, M25_off, M25_on, M26_off, M26_on, M27_off, M27_on, M28_off, M28_on, M29_off, M29_on, M30_off, M30_on, M31_off, M31_on, M32_off, M32_on, M33_off, M33_on, M34_off, M34_on, M35_off, M35_on, M36_off, M36_on, M37_off, M37_on, M38_off, M38_on, M39_off, M39_on, M40_off, M40_on, M41_off, M41_on, M42_off, M42_on, M43_off, M43_on, M44_off, M44_on, M45_off, M45_on, M46_off, M46_on, M47_off, M47_on, M48_off, M48_on, M49_off, M49_on, M50_off, M50_on, M51_off, M51_on, Person, Work' 
    dataset_name = "Tulum2010"
    #dataset_name = "Twor2009"

    
    print(dataset_name)
    address_for_save = r"E:\pgmpy\{}\sensor_maps.csv".format(dataset_name)

    #create_map_of_sensors(dataset_name = dataset_name, has_header = False, address_for_save = address_for_save, isSave = True, header = eval("file_header_" + dataset_name))# file_header_Tulum2009)
    #read_pandas_dataframe_from_csv(address_for_save)
    #test_check_the_neighberhood_of_two_sensors(address_for_save, 'M49', 'M36')
    test_percent_of_occlusion(dataset_name)