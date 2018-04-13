'''
Created on Apr 13, 2018

@author: Adele
'''
import csv 
import numpy as np

def read_sequence_based_CSV_file(file_address , has_header):
    '''
    Parameters:
    ==========
    file_address:
    has_header:
    
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
            list_of_persons[line] = int(other[1])
            list_of_activities[line] = int(other[2])

            list_of_data[line] = seq
        '''print(type(list_of_data[1]))
        for i in range(3):
            print(type(list_of_data[1][i]))
        '''
    
    return list_of_data , list_of_persons , list_of_activities

if __name__ == "__main__":
    #print(type([1,2]))
    list_of_data , list_of_persons , list_of_activities = read_sequence_based_CSV_file(file_address = r"E:\a1.csv", has_header = True)
    print(len(list_of_data) , '\n' ,len( list_of_persons) ,'\n' ,len(list_of_activities))