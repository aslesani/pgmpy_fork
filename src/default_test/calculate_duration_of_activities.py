'''
Created on Feb 4, 2019

@author: Adele
'''

from dataPreparation import get_list_of_allowed_sensors_and_works_in_dataset, get_person_and_work, convert_string_to_datetime
from datetime import timedelta

def get_list_of_duration_of_activities(file_address):
    
    f = open(file_address,"r")
    _, _, list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(file_address)
    
    list_of_beginned_activities = []# list of dictionaries, each cell contains the activity and personID of beginner
    number_of_proccessed_line = -1
    list_of_durations = []
    PersonNumber, WorkNumber = -1, -1
    
    for line in f:
        
        number_of_proccessed_line +=1
        cells = line.split()
        
        if len(cells) > 4:
            PersonNumber, WorkString = get_person_and_work(cells[4])
            WorkNumber = list_of_works.index(WorkString)
             
            if cells[5] == 'begin':
                list_of_beginned_activities.append([PersonNumber, WorkNumber, cells[0] , cells[1]])
                    
            elif cells[5] == 'end':
                
                indices = [i for i, (first, second, *_) in enumerate(list_of_beginned_activities) if (first, second) == (PersonNumber, WorkNumber)]
                print(indices)
                index = indices[0]
                begin_datetime = convert_string_to_datetime(list_of_beginned_activities[index][2], list_of_beginned_activities[index][3])
                end_datetime = convert_string_to_datetime(cells[0],cells[1])
                timedelta_in_sec = timedelta.total_seconds(end_datetime - begin_datetime)
                list_of_durations.append(timedelta_in_sec)
                del(list_of_beginned_activities[index])

            
    return list_of_durations

if __name__ == '__main__':

    file_address_Twor2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
    file_address_Tulum2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2010\data_edited by adele"
    file_address_Tulum2009 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\tulum2009\data.txt"
    file_address_Twor2010 = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\9 tulum\twor.2010\data_edited_by_adele"
    Test = r"E:\pgmpy\Test\annotated"
    list_of_activities = get_list_of_duration_of_activities(file_address_Twor2009)
    print(len(list_of_activities))
