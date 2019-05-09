# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:20:20 2018

@author: Adele
"""
from dataPreparation import get_list_of_allowed_sensors_and_works_in_dataset


def create_header_string(a, add_work_col = True):
    
    b = []
    header_string = ""
    for i in a:
        b.append(i + "_on")
        b.append(i + "_off")
        header_string = header_string + str(i) + "_on,"
        header_string = header_string + str(i) + "_off,"
    
    b.append('Person')
    header_string = header_string + "Person,"
    
    if add_work_col:
        b.append('Work')
        header_string = header_string + "Work"
    
    print(b)
    print(len(b))
    print(header_string)
    
    return header_string
    
if __name__ == "__main__":
    
    twor2009 = ["M01", "M02", "M03" , "M04" ,"M05" , "M06" , "M07","M08", "M09", "M10" , "M11" ,"M12", "M13", "M14" , "M15" ,"M16" , "M17" , "M18" , "M19" , "M20", "M21", "M22", "M23", "M24" , "M25" ,"M26" , "M27" , "M28" , "M29" , "M30", "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40", "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50", "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14", "D15"]
    Tulum2009 = ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016']
    Tulum2010 = ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031']
    twor2010 = ['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'I006', 'I010', 'I011', 'I012', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M036', 'M037', 'M038', 'M039', 'M040', 'M041', 'M042', 'M043', 'M044', 'M045', 'M046', 'M047', 'M048', 'M049', 'M050', 'M051']
    
    Test = r"E:\pgmpy\Test\annotated"

    list_of_Test_sensors, number_of_allowed_samples, list_of_works = get_list_of_allowed_sensors_and_works_in_dataset(Test)
    print(list_of_Test_sensors)
    a = create_header_string(list_of_Test_sensors)
    print(a)