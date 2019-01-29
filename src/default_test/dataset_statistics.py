# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 1:44:57 2019

@author: Adele
"""

def get_dataset_statistics(file_address):
    
    '''
   
    
    '''
    set_of_whole_sesnors = set()
    set_of_binary_sesnors = set()

    set_of_works = set()
    set_of_persons = set()
    number_of_binary_sensor_events = 0
    number_of_whole_sensor_events = 0
    
    f = open( file_address ,"r")

    for line in f:
        number_of_whole_sensor_events += 1
        cells = line.split()
        set_of_whole_sesnors.add(cells[2])
        #print(cells)
        #try:
        if cells[2][0] in ['M','I','D']:
            
            #if the sensor is not a float number
            #replace . by '', becasue of numbers like '22.5'
            if cells[3].replace('.','').isdigit() == False:
                
                set_of_binary_sesnors.add(cells[2])
                number_of_binary_sensor_events +=1
            
                if len(cells) > 4:
                    if cells[4][0] != 'R':
                        set_of_works.add(cells[4])
                    else:
                        set_of_works.add(cells[4][3:])
                        set_of_persons.add(cells[4][1])
    
    print("set_of_whole_sesnors:", set_of_whole_sesnors)    
    print("set_of_binary_sesnors:", set_of_binary_sesnors)                
    print("number of whole sensors:" , len(set_of_whole_sesnors))
    print("number of binary sensors:" , len(set_of_binary_sesnors))

    print("set_of_works:", set_of_works)
    print("number of works:" , len(set_of_works))                
    print("set_of_persons:", set_of_persons) 

    print("number_of_binary_sensor_events:", number_of_binary_sensor_events)             
    print("number_of_whole_sensor_events:", number_of_whole_sensor_events)             
            
            
         
 
if __name__ == "__main__":

    file_address_Towr2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
    file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
    file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"
    file_address_Twor2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\twor.2010\data"
   
    get_dataset_statistics(file_address_Tulum2010)