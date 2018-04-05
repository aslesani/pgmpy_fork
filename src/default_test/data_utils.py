# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 01:35:57 2018

@author: Adele
"""
import pandas as pd
import numpy as np

def check_data(data1 , data2 , remove_latent_variables):
    '''
    check the data to understand if there is a variable that exists in test but not in train set
    if True, remove that row of data if desired
    
    Parameters:
    ==========
    data1 , data2: numpy ndarray with same number of columns. 
                   if they are pandas dataframe, convert them to numpy, process them and then retrun as pandas dataframe
    remove_latent_variables: if True, remove the rows with latent variables
    
    Returns:
    ========
    are_different: True if there is at least one difference
    corrected_data: if remove_latent_variables == True, retruns the corrected data
    '''
    if type(data1) == pd.core.frame.DataFrame:
        data1 = data1.values
    
    is_data2_pd = False
    if type(data2) == pd.core.frame.DataFrame:
        is_data2_pd = True
        data2_columns = data2.columns
        data2 = data2.values
        
    r1 , c1 = np.shape(data1)
    r2 , c2 = np.shape(data2)
    
    if c1 != c2:
        raise TypeError("both data1 and data2 should have the same number of columns")
    
    are_different = False
    
    for c in range(c1):
        
        set_of_values_in_data1 = set(data1[: , c])
        set_of_values_in_data2 = set(data2[: , c])
        diffrences = set_of_values_in_data2 - set_of_values_in_data1
        
        if diffrences != set():
            are_different = True
            if remove_latent_variables == False:
                return True , ()
            else:
                for item in diffrences:
                    my_ind = np.where(np.equal(data2[: , c] , item ))
                    data2 = np.delete(data2 , my_ind , axis = 0)

    
    if is_data2_pd:
        data2 = pd.DataFrame(data2 , columns = data2_columns)
        
    return are_different , data2             
            
if __name__ == "__main__":
    
    a = pd.DataFrame([[1,2,3] , [1,2,5]])
    b = pd.DataFrame([[1,2,7] , [1,2,3] , [1,2,7] , [1,2,8]])
    
    are_d , data2 = check_data(a , b,  True)
    print(are_d)
    print(data2)
