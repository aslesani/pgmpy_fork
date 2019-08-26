'''
Created on Aug 26, 2019

@author: Adele
'''
import re
import numpy as np
openshs_activities = {"sleep":0, "eat":1, "personal":2, "work":3, "wakeup":4, "watchTV":5, "other":6, "cook":7}
from dataPreparation import convert_string_to_datetime


def get_number_of_lines_in_file(fname, hasHeader):
    '''
	if the file has a header, the count = count -1
	'''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    
    if hasHeader:
        return i
	
    return i + 1
	
	
def convert_data_to_each_row_one_feature_is_on(in_file, out_file, person_number, hasHeader):
    '''
	this function adds the person number to file as well
	there is a mistake that this code ignores the first activity if the activity does not change any sesnor status 
	'''
    f = open(in_file,"r")
    #read the first line to get the sensor names
    seprators_pattern = ',| |\n'
    first_line = re.split(seprators_pattern,f.readline())# the last element should be ''. do ont count it
    
	#-3 for Activity,timestamp,''
	#+5 for Person + work + 1 date + 1 time + 1 datetime for ordering data
    number_of_columns = (len(first_line) - 3) * 2 + 5 
    number_of_sensors_plus_activity = len(first_line) - 2
    list_of_sensor_names = first_line[0:(number_of_sensors_plus_activity - 1)]
    features = [0] * number_of_columns 
    features[-5] = person_number
   
    number_of_samples = get_number_of_lines_in_file(in_file, hasHeader)
    all_features = np.zeros((number_of_samples, number_of_columns), dtype= object)
    
    counter = -1
    #first = True
    previous_cells = re.split(seprators_pattern,f.readline()) #first line of samples
	
    for line in f:
        #search for the first change in sensor states
        cells = re.split(seprators_pattern, line)
        if cells[0:number_of_sensors_plus_activity] != previous_cells[0:number_of_sensors_plus_activity]:
            counter +=1
			#find the sensor whose status is changed
            for i in range(number_of_sensors_plus_activity-1):
                if cells[i] != previous_cells[i]:
                    if cells[i] == '1':
                        changed_index = i*2
                    elif cells[i] == '0':
                        changed_index = i*2 + 1
                    break
					
            #set_of_changed_index.add(changed_index)
            features[changed_index] = 1
            features[-4] = openshs_activities[cells[-4]]
            features[-3] = cells[-3]
            features[-2] = cells[-2]
            features[-1] = convert_string_to_datetime(cells[-3],cells[-2])

            previous_cells = cells
            
            if counter < number_of_samples:
                all_features[counter] = features
            else:
                all_features = np.vstack([all_features,features])
            
            #reset changed_index to 0
            features[changed_index] = 0
			
    all_features = np.delete(all_features , range(counter + 1 , number_of_samples) , axis = 0)    
    #print(all_features[:,-1])
    all_features = all_features[all_features[:,-1].argsort()] # sort all_features based on datetime column

    rows, cols = all_features.shape
    print("rows:", rows, "cols:" , cols)
    
    np.savetxt(out_file, np.delete(all_features, -1 , 1 ), delimiter=',' , fmt='%s')
				
					
if __name__ == "__main__":
    
    base_address = r"E:\Lessons_tutorials\Behavioural user profile articles\openSHS\openshs-master-new\app\datasets"
    person_name = input("Enter Person Name:")#"test"
    number_of_days = int(input("Enter number of days:"))#1
    in_file = base_address + "\{p}_{nd}days.csv".format(p = person_name, nd = number_of_days)
    out_file = base_address + "\{p}_{nd}days_each_row_one_features_is_one_on_and_off.csv".format(p = person_name, nd = number_of_days)
   
    person_ID = int(input("Enter ID of person:"))
    convert_data_to_each_row_one_feature_is_on(in_file, out_file, person_ID, True)

