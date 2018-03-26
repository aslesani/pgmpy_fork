# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:43:11 2018

@author: Adele
"""

from os import listdir
from os.path import isfile, join
import numpy as np

def remove_first_line_from_file(filename):
    with open(filename, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(filename, 'w') as fout:
        fout.writelines(data[1:])

        
if __name__ == "__main__":
    #line_prepender(r"E:\test2.csv" , "# added by... :D")
    
    
    no_overlap = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_no overlap_based on different deltas\{t}"
    activity = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_based on activities\{t}"
    activity_delta = r"E:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\{t}"
    
    dirs = [no_overlap.format(t = t) for t in ['train' , 'test']]
    header1 = [activity.format(t = t) for t in ['train' , 'test']]
    header2 = [activity_delta.format(t=t) for t in ['train' , 'test']]
    dirs = np.concatenate((dirs , header1 , header2) , axis = 0)
    
    for mypath in dirs: 
       list_of_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
       print(list_of_files)
       print("*****************")
       for f in list_of_files:
           print(f)
           remove_first_line_from_file(join(mypath, f))

    