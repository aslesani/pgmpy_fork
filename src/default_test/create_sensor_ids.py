# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:20:20 2018

@author: Adele
"""

a = ["M01", "M02", "M03" , "M04" ,"M05" , "M06" , "M07","M08", "M09", "M10" , "M11" ,"M12", "M13", "M14" , "M15" ,"M16" , "M17" , "M18" , "M19" , "M20", "M21", "M22", "M23", "M24" , "M25" ,"M26" , "M27" , "M28" , "M29" , "M30", "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40", "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50", "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14", "D15"]
print(len(a))
b = []
header_string = ""
for i in a:
     b.append(i + "_on")
     b.append(i + "_off")
     header_string = header_string + str(i) + "_on,"
     header_string = header_string + str(i) + "_off,"

b.append('Person')
b.append('Work') 
header_string = header_string + "Person,"
header_string = header_string + "Work"


    
print(b)
print(len(b))

print(header_string)