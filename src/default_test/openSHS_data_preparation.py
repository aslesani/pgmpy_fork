'''
Created on Aug 26, 2019

@author: Adele
'''

def convert_data_to_each_row_one_feature_is_on(in_file, out_file):
    pass
				
					
if __name__ == "__main__":
    
    base_address = r"E:\Lessons_tutorials\Behavioural user profile articles\openSHS\openshs-master-new\app\datasets"
	person_name = "Adele"
	number_of_days = 30
    in_file = base_address + "\{p}_{nd}days.csv".format(p = person_name, nd = number_of_days)
	out_file = base_address + "\{p}_{nd}days_each_row_one_features_is_one_on_and_off.csv".format(p = person_name, nd = number_of_days)
	convert_data_to_each_row_one_feature_is_on(in_file, out_file)

