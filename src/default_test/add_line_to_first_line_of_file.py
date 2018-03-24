# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:55:46 2018

@author: Adele
"""
from os import listdir
from os.path import isfile, join
import numpy as np

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
        
if __name__ == "__main__":
    #line_prepender(r"E:\test2.csv" , "# added by... :D")
    
    file_header = '# M01_on, M01_off, M02_on, M02_off, M03_on, M03_off, M04_on, M04_off, M05_on, M05_off, M06_on, M06_off, M07_on, M07_off, M08_on, M08_off, M09_on, M09_off, M10_on, M10_off, M11_on, M11_off, M12_on, M12_off, M13_on, M13_off, M14_on, M14_off, M15_on, M15_off, M16_on, M16_off, M17_on, M17_off, M18_on, M18_off, M19_on, M19_off, M20_on, M20_off, M21_on, M21_off, M22_on, M22_off, M23_on, M23_off, M24_on, M24_off, M25_on, M25_off, M26_on, M26_off, M27_on, M27_off, M28_on, M28_off, M29_on, M29_off, M30_on, M30_off, M31_on, M31_off, M32_on, M32_off, M33_on, M33_off, M34_on, M34_off, M35_on, M35_off, M36_on, M36_off, M37_on, M37_off, M38_on, M38_off, M39_on, M39_off, M40_on, M40_off, M41_on, M41_off, M42_on, M42_off, M43_on, M43_off, M44_on, M44_off, M45_on, M45_off, M46_on, M46_off, M47_on, M47_off, M48_on, M48_off, M49_on, M49_off, M50_on, M50_off, M51_on, M51_off, I03_on, I03_off, D03_on, D03_off, D05_on, D05_off, D07_on, D07_off, D08_on, D08_off, D09_on, D09_off, D10_on, D10_off, D12_on, D12_off, D14_on, D14_off, D15_on, D15_off, Person, Work'
    
    no_overlap = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_no overlap_based on different deltas\{t}"
    activity = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_based on activities\{t}"
    activity_delta = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\{t}"
    
    header_without_work_labels = [no_overlap.format(t = t) for t in ['train' , 'test']]
    header1 = [activity.format(t = t) for t in ['train' , 'test']]
    header2 = [activity_delta.format(t=t) for t in ['train' , 'test']]
    header_with_work_labels = np.concatenate((header1 , header2) , axis = 0)
    
    '''
    for mypath in header_without_work_labels: 
        list_of_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print(list_of_files)
        print("*****************")
        for f in list_of_files:
            print(f)
            line_prepender(join(mypath, f) , file_header.replace(', Work' , ''))
'''
    for mypath in header_with_work_labels: 
            list_of_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print(list_of_files)
            print("*****************")
            for f in list_of_files:
                print(f)
                line_prepender(join(mypath, f) , file_header)

 #       line_prepender(activity.format(t) , file_header)
#        line_prepender(activity_delta.format(t) , file_header)
        
    
    
    