'''
Created on Aug 24, 2019

@author: Adele
'''
import csv
import os


def remove_blank_lines(in_filename):
        
    temp_filename = r"E:\Lessons_tutorials\Behavioural user profile articles\openSHS\openshs-master-new\app\datasets\dataset2.csv"

    with open(in_filename) as in_file:
        with open(temp_filename, 'w') as out_file:
            writer = csv.writer(out_file)
            for row in in_file.readlines():
                if row.strip() != '':
                    out_file.write(row)
    
    os.remove(in_filename)	
    os.rename(temp_filename, in_filename)	
					
					
if __name__ == "__main__":
    in_file = r"E:\Lessons_tutorials\Behavioural user profile articles\openSHS\openshs-master-new\app\datasets\dataset.csv"
    base_address = r"E:\Lessons_tutorials\Behavioural user profile articles\openSHS\openshs-master-new\app\datasets"
    person_name = "Maman"
    number_of_days = 30
    in_file = base_address + "\{p}_{nd}days.csv".format(p = person_name, nd = number_of_days)

    remove_blank_lines(in_file)