'''
Created on Apr 13, 2018

@author: Adele
'''
from dataPreparation import casas7_to_csv_time_Ordered, casas7_to_csv_based_on_each_person_sensor_events_time_Ordered
from read_write import read_data_from_CSV_file

if __name__ == "__main__":
  
    #casas7_to_csv_time_Ordered()
    address_to_read = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\annotated"
    address_to_save = r'E:\pgmpy\sensor_data_each_person_sensor_events+time_ordered.csv'
    #casas7_to_csv_based_on_each_person_sensor_events_time_Ordered(address_to_read, address_to_save , 2)
    a = read_data_from_CSV_file(dest_file = address_to_save , data_type = int ,  has_header = False , return_as_pandas_data_frame = False , remove_date_and_time = True , return_header_separately = False , convert_int_columns_to_int = True)
    print(a)
    print(a.shape)