# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:42:37 2018

@author: Adele
"""
import numpy as np
import csv

dest = r'E:\test.csv'
np.savetxt(dest, 
                [[1,2,3 , 4,5,6] , [7,8,9, 4,5,6]], delimiter=',' , fmt='%s' , header = "a1,a2,a3, a4 , a5,a6")


with open(dest,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
        header = next(data_iter)
        header[0] = header[0].split('# ')[1]
        print("header:" , header)
        for data in data_iter:
            print(data)

