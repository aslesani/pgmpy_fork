'''
Created on Apr 13, 2018

@author: Adele
'''
import csv 
import numpy as np

def read_sequence_based_CSV_file_with_activity(file_address , has_header, separate_data_based_on_persons):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
    list_of_activities = list([0] * number_of_rows)            
       
    for line in range(0 , number_of_rows):
        d = list_of_data[line].split('[')[1]
        d = d.split(']')
        seq = d[0].replace("'" , '').split(', ')
        other = d[1].split(',')
        list_of_persons[line] = int(other[1])# * len(seq)
        list_of_activities[line] = int(other[2])
        list_of_data[line] = np.array(seq)
   
    if separate_data_based_on_persons:
        
        number_of_persons = len(set(list_of_persons))
        new_list_of_data = np.zeros(number_of_persons , dtype = np.ndarray)
        new_list_of_persons = np.zeros(number_of_persons ,dtype = np.ndarray)
        new_list_of_activities = np.zeros(number_of_persons ,dtype = np.ndarray)
        
        list_of_persons = np.array(list_of_persons)
        list_of_data = np.array(list_of_data)
        list_of_activities = np.array(list_of_activities)
        
        new_list_index = 0
        for per in list(sorted(set(list_of_persons))):
            indexes = np.where(np.equal(list_of_persons , per))
            new_list_of_persons[new_list_index] = list_of_persons[indexes]
            new_list_of_data[new_list_index] = list_of_data[indexes]
            new_list_of_activities[new_list_index] = list_of_activities[indexes]
            new_list_index += 1
            
        return new_list_of_data , new_list_of_persons , new_list_of_activities
    
    else:
        return list_of_data , list_of_persons , list_of_activities



def read_sequence_based_CSV_file_without_activity(file_address , has_header, separate_data_based_on_persons):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
       
    for line in range(0 , number_of_rows):
        d = list_of_data[line].split('[')[1]
        d = d.split(']')
        seq = d[0].replace("'" , '').split(', ')
        other = d[1].split(',')
        list_of_persons[line] = int(other[1])# * len(seq)
        list_of_data[line] = np.array(seq)
   
    if separate_data_based_on_persons:
        
        number_of_persons = len(set(list_of_persons))
        new_list_of_data = np.zeros(number_of_persons , dtype = np.ndarray)
        new_list_of_persons = np.zeros(number_of_persons ,dtype = np.ndarray)
        
        list_of_persons = np.array(list_of_persons)
        list_of_data = np.array(list_of_data)
        
        new_list_index = 0
        for per in list(sorted(set(list_of_persons))):
            indexes = np.where(np.equal(list_of_persons , per))
            new_list_of_persons[new_list_index] = list_of_persons[indexes]
            new_list_of_data[new_list_index] = list_of_data[indexes]
            new_list_index += 1
            
        return new_list_of_data , new_list_of_persons
    
    else:
        return list_of_data , list_of_persons


def read_sequence_of_bags_CSV_file_with_activity(file_address , has_header, separate_data_based_on_persons):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    separate_data_based_on_persons: if True, retrun return values as a numpy array with len of number of persons
    
    Retrun:
    =======
    list_of_data: list of sequenced data
    list_of_persons: list of corresponding person numbers
    list_of_activities: list of corresponding activity numbers

    '''
    header = ""
    list_of_data = []
    with open(file_address,'r') as dest_f:
        list_of_data = list(dest_f)
        number_of_rows = len(list_of_data)
        if has_header:
            number_of_rows -= 1
            header = list_of_data[0].split('# ')[1] 
            list_of_data = list_of_data[1:]#remove the first line(header)
        
    list_of_persons = list([0] * number_of_rows) 
    list_of_activities = list([0] * number_of_rows)            
       
    for line in range(0 , number_of_rows):
        #print('list_of_data[line]:' , list_of_data[line])
        d = list_of_data[line].split('[[')[1]
        #print('d1:' , d)
        #print('type(d):' , type(d))

        d = d.split(']]')
        #print('len(d2):' , len(d))
        #d = d.split(']')
        d[0] = d[0].split('], [')
        #print('len(d[0]):' , len(d[0]))
        seq = list([0] * len(d[0]))
        for i in range(len(d[0])):
            seq[i] = d[0][i].split(', ')
        #print('seq:' , len(seq) , type(seq) , type(seq[0]) , len(seq[0]) , type(seq[0][0]))
        other = d[1].split(',')
        #print('other:' , other)
        list_of_persons[line] = int(other[1])# * len(seq)
        #print('list_of_persons[line]:' , list_of_persons[line])
        list_of_activities[line] = int(other[2])
        #print('list_of_activities[line]:' , list_of_activities[line])
        for i in range(len(seq)):
            for j in range(len(seq[i])):
                seq[i][j] = int(seq[i][j])
        
        list_of_data[line] = np.array(seq)
        
    
    if separate_data_based_on_persons:
        
        number_of_persons = len(set(list_of_persons))
        new_list_of_data = np.zeros(number_of_persons , dtype = np.ndarray)
        new_list_of_persons = np.zeros(number_of_persons ,dtype = np.ndarray)
        new_list_of_activities = np.zeros(number_of_persons ,dtype = np.ndarray)
        
        list_of_persons = np.array(list_of_persons , dtype = np.ndarray)
        list_of_data = np.array(list_of_data)
        list_of_activities = np.array(list_of_activities)
        
        new_list_index = 0
        for per in list(sorted(set(list_of_persons))):
            indexes = np.where(np.equal(list_of_persons , per))
            new_list_of_persons[new_list_index] = list_of_persons[indexes]
            new_list_of_data[new_list_index] = list_of_data[indexes]
            new_list_of_activities[new_list_index] = list_of_activities[indexes]
            new_list_index += 1
            
        return new_list_of_data , new_list_of_persons , new_list_of_activities
    
    else:
        return list_of_data , list_of_persons , list_of_activities


def repaet_person_tags_as_much_as_seq_length(list_of_data , list_of_persons, is_one_person):
   
    if is_one_person:
        for i in range(len(list_of_data)):
            list_of_persons[i] = list([list_of_persons[i]] * len(list_of_data[i]))
            
        
    else:
        number_of_persons = len(list_of_persons)
        print("number_of_persons:" , number_of_persons)
        for per in range(number_of_persons):
            for i in range(len(list_of_data[per])):
                a = np.array(list([list_of_persons[per][i]] * len(list_of_data[per][i])))
                
                list_of_persons[per][i] = a
   
    return list_of_persons.copy()

if __name__ == "__main__":
    #print(type([1,2]))
    #list_of_data , list_of_persons , list_of_activities = read_sequence_based_CSV_file(file_address = r"E:\a1.csv", has_header = True , separate_data_based_on_persons=True)
    #print("list_of_data:" , list_of_data[0][20] )
    #print("len(list_of_data):" , len(list_of_data))
    #print("list_of_persons:" , list_of_persons[0][0] )
    #print("len(list_of_persons):" , len(list_of_persons))
    #print("list_of_activities:" , list_of_activities )
    #print("len(list_of_activities):" , len(list_of_activities))
    separate_data_based_on_persons = True
    a , b , c = read_sequence_of_bags_CSV_file_with_activity(file_address = r'C:\b.csv' , has_header= True, separate_data_based_on_persons = separate_data_based_on_persons)
    #print(a)
    #print("###########################################")
    #print(b)
    repeated_per = repaet_person_tags_as_much_as_seq_length(a , b , separate_data_based_on_persons=separate_data_based_on_persons)
    print(type(repeated_per[0][0]))
    #print("###########################################")

    #print(b)
    
    
    
    #print(len(list_of_data) , '\n' ,len( list_of_persons) ,'\n' ,len(list_of_activities))