'''
Created on May 14, 2017

@author: Adele
'''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import csv
import time
#from numpy import dtype
#from h5py._hl.datatype import Datatype
import os.path
from numpy.core.numeric import False_
from numpy import string_
#from default_test.parameter_learning_of_Aras_data import prior_type

feature_names = ["M01", "M02", "M03", "M04" , "M05" , "M06" , "M07" , "M08" , "M09" , "M10"
                       , "M11", "M12", "M13", "M14" , "M15" , "M16" , "M17" , "M18" , "M19" , "M20"
                       , "M21", "M22", "M23", "M24" , "M25" , "M26" , "M27" , "M28" , "M29" , "M30"
                       , "M31", "M32", "M33", "M34" , "M35" , "M36" , "M37" , "M38" , "M39" , "M40"
                       , "M41", "M42", "M43", "M44" , "M45" , "M46" , "M47" , "M48" , "M49" , "M50"
                       , "M51", "I03", "D03", "D05" , "D07" , "D08" , "D09" , "D10" , "D12" , "D14"
                       , "D15", "PNo", "WNo", "Date" , "Time"]
dest_file = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor_data+time_ordered.csv'   

base_address = r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\ '

def PCA_data_generation(file_address,base_address_to_save, remove_date_and_time , remove_activity_column):
    '''
    Parameter:
    =========
    file_address:
    base_address_to_save:
    remove_date_and_time: if is true, the time and date columns are removed
    remove_activity_column: if is true, the activity/work column is removed
    '''
    
    sensor_data = read_data_from_file(file_address, np.int, remove_date_and_time)
    print(sensor_data)#:, -1])

    
    if remove_activity_column == True:
        sensor_data = np.delete(sensor_data ,-1 , 1)
    
    rows , cols = np.shape(sensor_data)
    print("file address: " , file_address)
    print("cols={}".format(cols))
    target = np.zeros((rows, 1), dtype= int )
    for ind in range(rows):
        target[ind][0] = sensor_data[ind,-1] # person number is considered as class
   
    print(target)
    sensor_data = np.delete(sensor_data ,-1 , 1) # remove the Person column
    for i in range(2,41):#cols):
        pca = PCA(n_components=i)
        
        
        data_new = pca.fit_transform(sensor_data) #Fit the model with X and apply the dimensionality reduction on X.
        # Ù¾Ø³ Ù�ÛŒØª Ù�Ù‚Ø· Ø§Ø¹Ù…Ø§Ù„ Ù…ÛŒ Ú©Ù†Ø¯ ÙˆÙ„ÛŒ Ø±ÙˆÛŒ Ø¯Ø§Ø¯Ù‡ Ú©Ø§Ø±ÛŒ Ù†Ù…ÛŒ Ú©Ù†Ø¯. Ø¸Ø§Ù‡Ø±Ø§ Ù�Ù‚Ø· Ù…Ù‚Ø§Ø¯ÛŒØ± ÙˆÛŒÚ˜Ù‡ Ø±Ø§ Ù…ÛŒ ÛŒØ§Ø¨Ø¯. 
        #Ù�ÛŒØª ØªØ±Ù†Ø³Ù�Ø±Ù… Ù‡Ù… Ø¢Ù† Ú©Ø§Ø± Ø±Ø§ Ø§Ù†Ø¬Ø§Ù… Ù…ÛŒ Ø¯Ù‡Ø¯ Ù‡Ù… Ø±ÙˆÛŒ Ø¯Ø§Ø¯Ù‡ Ø§Ø¹Ù…Ø§Ù„ Ù…ÛŒ Ú©Ù†Ø¯ Ùˆ Ø¯Ø§Ø¯Ù‡ Ú©Ø§Ù‡Ø´ Ø¨Ø¹Ø¯ Ù¾ÛŒØ¯Ø§ Ù…ÛŒ Ú©Ù†Ø¯
        # Ø¨Ø¹Ø¯ Ø§Ø² Ø¢Ù† Ø¨Ø±Ø§ÛŒ Ø¯Ø§Ø¯Ù‡ Ù‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ Ù�Ù‚Ø· Ø¨Ø§ÛŒØ¯ ØªØ±Ù†Ø³Ù�Ø±Ù… Ø±Ùˆ Ú©Ø§Ù„ Ú©Ù†ÛŒÙ…. Ú†ÙˆÙ† Ù‚Ø¨Ù„Ø§ Ù…Ø¯Ù„ Ø³Ø§Ø®ØªÙ‡ Ø´Ø¯Ù‡ Ø§Ø³Øª Ù�Ù‚Ø· Ø¨Ø§ÛŒØ¯ Ø¯Ø§Ø¯Ù‡ Ù‡Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ Ø±Ø§ Ø¨Ù‡ Ù�Ø¶Ø§ÛŒ Ø¬Ø¯ÛŒØ¯ Ø¨Ø¨Ø±Ø¯
        print(data_new.shape)
        print(target.shape)
        dest = base_address_to_save + 'PCA_n=' + str(i) +'.csv'
        print(dest)
        
        np.savetxt(dest, np.concatenate((data_new, target), axis=1), delimiter=',' , fmt='%s')
    

def PCA_data_generation_on_separated_train_and_test(file_address,base_address_to_save, remove_date_and_time , remove_activity_column, test_file_address, base_address_of_test_file_to_save):
    '''
    Parameter:
    =========
    file_address:
    base_address_to_save:
    remove_date_and_time: if is true, the time and date columns are removed
    remove_activity_column: if is true, the activity/work column is removed
    '''
    
    sensor_data = read_data_from_file(file_address, np.int, remove_date_and_time)
    test_data = read_data_from_file(test_file_address, np.int, remove_date_and_time)
    print(sensor_data)#:, -1])

    
    if remove_activity_column == True:
        sensor_data = np.delete(sensor_data ,-1 , 1)
        test_data = np.delete(test_data ,-1 , 1)
    
    train_rows , train_cols = np.shape(sensor_data)
    train_target = np.zeros((train_rows, 1), dtype= int )
    
    for ind in range(train_rows):
        train_target[ind][0] = sensor_data[ind,-1] # person number is considered as class
   
   
    test_rows , test_cols = np.shape(test_data)
    test_target = np.zeros((test_rows, 1), dtype= int )
    for ind in range(test_rows):
        test_target[ind][0] = test_data[ind , -1]
    
    
    sensor_data = np.delete(sensor_data ,-1 , 1) # remove the Person column
    test_data = np.delete(test_data ,-1 , 1) # remove the Person column

    for i in range(2,41):#cols):
        pca = PCA(n_components=i)
        
        
        train_data_new = pca.fit_transform(sensor_data) #Fit the model with X and apply the dimensionality reduction on X.
        print(train_data_new.shape)
        print(train_target.shape)
        dest = base_address_to_save + 'PCA_n=' + str(i) +'.csv'
        print(dest)
        
        np.savetxt(dest, np.concatenate((train_data_new, train_target), axis=1), delimiter=',' , fmt='%s')
        
        #transform test data
        print(sensor_data.shape)
        print(test_data.shape)
        test_data_new = pca.transform(test_data)
        test_dest = base_address_of_test_file_to_save + 'PCA_n=' + str(i) +'.csv'
        print(test_dest)
        
        np.savetxt(test_dest, np.concatenate((test_data_new, test_target), axis=1), delimiter=',' , fmt='%s')

        


def create_BN_model(data): 
    '''
    create a Bayesian Model (structure learning and Parameter learning)
    
    Parameters:
    ==========
    data: the data which we want to create a bn for
    
    Returns:
    =======
    model:
    learning_time:
    
    '''   
    #data = pd.DataFrame(sensor_data)#, columns= feature_names)#['X', 'Y'])
    #print("&&&&&&&&&&&&&&&&&")
    #print(data)
    #data = pd.DataFrame(data)#read_data_from_file_remove_date_and_time(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor+PCA_n=5.csv" , data_type='float'))
    #print("2222222222222222")
    #print(type(data))

    #print(data)
    
    #start_time = time.time()
    # 2 hours running, without output
    #hc = HillClimbSearch(data, scoring_method=BicScore(data))
    #best_model = hc.estimate()
    #print(hc.scoring_method)
    #print(best_model.edges())
    #end_time = time.time()
    #print("execution time in seconds:")
    #print(end_time-start_time)
    
    #start_time = time.time()
    #hc = HillClimbSearch(data, scoring_method=BdeuScore(data))
    #best_model = hc.estimate()
    #print(hc.scoring_method)
    #print(best_model.edges())
    #end_time = time.time()
    #print("execution time in seconds:")
    #print(end_time-start_time)
    
    #structure learning
    print("structure learning")
    start_time = time.time()
    hc = HillClimbSearch(data, scoring_method= BicScore(data))#K2Score(data))#BicScore(data))#K2Score(data))BdeuScore(data)
    best_model = hc.estimate()
    #print(hc.scoring_method)
    print(best_model.edges())
    end_time = time.time()
    sl_time = end_time - start_time
    print("execution time in seconds:{}".format(sl_time))

    #parameter learning
    #model = BayesianModel([('A', 'C'), ('B', 'C')])
    #model.fit(data)
    #model.get_cpds()

    ######
    #best_model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])

    start_time = time.time()
    casas7_model = BayesianModel(best_model.edges())
    #print("*******************")
    #BayesianEstimator.get_parameters(self, prior_type, equivalent_sample_size, pseudo_counts)
    #####estimator = BayesianEstimator(best_model, data)
    #####print(estimator.get_parameters(prior_type='K2'))#, equivalent_sample_size=5)
    
    casas7_model.fit(data, estimator=BayesianEstimator, prior_type="K2")#MaximumLikelihoodEstimator)
    end_time = time.time()
    ######print(casas7_model.get_cpds())
    ###casas7_model.predict(data)
    #print("casas7_model.node:{}".format(casas7_model.node))
    pl_time = end_time - start_time
    ########return estimator
    #for cpd in casas7_model.get_cpds():
     #   print(cpd)
        
    return (casas7_model , sl_time + pl_time)

    
def create_BN_model_using_BayesianEstimator(data):    
    #data = pd.DataFrame(sensor_data)#, columns= feature_names)#['X', 'Y'])
    #print(data)
    data = pd.DataFrame(data)#read_data_from_file_remove_date_and_time(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\sensor+PCA_n=5.csv" , data_type='float'))
    #print(data)
    
    #start_time = time.time()
    # 2 hours running, without output
    #hc = HillClimbSearch(data, scoring_method=BicScore(data))
    #best_model = hc.estimate()
    #print(hc.scoring_method)
    #print(best_model.edges())
    #end_time = time.time()
    #print("execution time in seconds:")
    #print(end_time-start_time)
    
    #start_time = time.time()
    #hc = HillClimbSearch(data, scoring_method=BdeuScore(data))
    #best_model = hc.estimate()
    #print(hc.scoring_method)
    #print(best_model.edges())
    #end_time = time.time()
    #print("execution time in seconds:")
    #print(end_time-start_time)
    
    #structure learning
    print("structure learning")
    start_time = time.time()
    hc = HillClimbSearch(data, scoring_method= K2Score(data))#BicScore(data))#K2Score(data))BdeuScore(data)
    best_model = hc.estimate()
    print(hc.scoring_method)
    print(best_model.edges())
    end_time = time.time()
    print("execution time in seconds:{}".format(end_time-start_time))

    #parameter learning
    #model = BayesianModel([('A', 'C'), ('B', 'C')])
    #model.fit(data)
    #model.get_cpds()

    ######
    #best_model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])

    casas7_model = BayesianModel(best_model.edges())
    print("*******************")
    #BayesianEstimator.get_parameters(self, prior_type, equivalent_sample_size, pseudo_counts)
    #####estimator = BayesianEstimator(best_model, data)
    #####print(estimator.get_parameters(prior_type='K2'))#, equivalent_sample_size=5)
    
    estimator = BayesianEstimator(casas7_model, data)
    
    
    #casas7_model.fit(data, estimator=BayesianEstimator, prior_type="K2")#MaximumLikelihoodEstimator)
    ######print(casas7_model.get_cpds())
    ###casas7_model.predict(data)
    #print("casas7_model.node:{}".format(casas7_model.node))
    
    ########return estimator
    return estimator

def read_data_from_file(dest_file, data_type , remove_date_and_time = True ):
    '''
    this function is used when there is data and time columns in dataset
    and the user want to remove them
    the output type is int
    
    Parameters:
    ==========
    dest_file:
    data_type:
    remove_date_and_time: if is true, the date and time columns(two last columns) are removed
    '''
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        data = [data for data in data_iter]
    
    if remove_date_and_time == True:
        data = np.delete(np.delete(data, -1, 1), -1 , 1)
    
    rows = len(data)
    print(rows)
    
    return_value= np.asarray(data, dtype = data_type)#np.int)
    #print(return_value)
    return return_value

def read_data_from_PCA_output_file(dest_file):
    '''
    this function is used to read PCA output files
    the output type of this function is float
    
    Parameters:
    ==========
    
    Returns:
    ========
    array: numpy 2d array
    
    '''
    print(dest_file)
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        data = [data for data in data_iter]
    
    
    return_value= np.asarray(data, dtype = np.float)
    return return_value
    
def read_data_from_PCA_digitized_file(dest_file):
    '''
    this function is used to read PCA digitized files
    the output type of this function is int
    
    Parameters:
    ==========
    
    Returns:
    =======
    array: panda dataframe which have column name. the last column is named "Person"
    
    '''
    with open(dest_file,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')#quotechar = '"')
    
        data = [data for data in data_iter]
    
    
    numpy_result = np.asarray(data, dtype = np.int)
    '''
    _ , cols = numpy_result.shape
    column_names = []

    for names in range(1, cols):
        column_names.append(str(names))
    column_names.append("Person")  
       
    panda_result = pd.DataFrame(data=numpy_result , columns= column_names , dtype = np.int) 
    
    print(panda_result.columns)
    #print(panda_result)
    
    return panda_result
    '''
    return numpy_result

def featureSelection_based_on_Variance(dest_file,threshold, remove_date_and_time = False , isSave , path_to_save , column_indexes_to_keep):
    '''suppose that we have a dataset with boolean features, and we want to remove all features that are either one or zero (on or off) in more than p%(e.g. 80%) of the samples. 
       Boolean features are Bernoulli random variables, and the variance of such variables is given by var[x] = p(1-p)
    
    Parameteres:
    ========
    dest_file
    threshold
    remove_date_and_time = False
    isSave
    path_to_save
    column_indexes_to_keep: the column indexes that want to be kept and the feature selection method does not apply on
    '''
    data = read_data_from_file(dest_file, np.int, remove_date_and_time=remove_date_and_time)
    #print(data)
    # remove person, work, date and time columns
    #sensor_data = np.delete(np.delete(np.delete(np.delete(data ,64 , 1), 63 , 1), 62 , 1), 61,1)

    rows , cols = data.shape
    print(rows , cols)
    column_indexes_to_apply_feature_selection = range(cols) - column_indexes_to_keep 

    #threshold=0.7 * (1 - 0.7)
    select_features = VarianceThreshold(threshold=threshold)# 80% of the data
    #select_features.
    data_new = select_features.fit(data[])
    print(data_new.shape)
    #print(sensor_data)
    four_last_columns = data[:, [-4,-3,-2,-1]]#np.select(data,[-1,-2,-3,-4])
    #print(four_last_columns)
    data_new = np.concatenate((data_new, four_last_columns), axis=1)
    #print(select_features.variances_)

    if(isSave):
        np.savetxt(path_to_save, data_new, delimiter=',' , fmt='%s')


def featureSelection_Kbest(k):
    '''
    Select features according to the k highest scores.

    '''
    data = read_data_from_file(dest_file, np.int, remove_date_and_time=True)
    #print(data)
    # remove person, work, date and time columns
    sensor_data = np.delete(np.delete(data ,62 , 1), 61 , 1)
    sensor_target = data[:, -2]###-1 ro felan alaki hazf kardam
    two_last_columns = data[:, [-2,-1]]#np.select(data,[-1,-2,-3,-4])
    #print(four_last_columns)
    

    print(sensor_data.shape)

    select_features = SelectKBest(chi2, k=k)
    #select_features.
    #print(sensor_target)
    
    sensor_data = select_features.fit_transform(sensor_data, sensor_target)
    print(sensor_data.shape)
    
    print(sensor_data)
    data_new = np.concatenate((sensor_data, two_last_columns), axis=1)
    #print(select_features.variances_)

    np.savetxt(r'E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy +\sensor_featureSelection_' + str(k) +'best.csv', 
                data_new, delimiter=',' , fmt='%s')


def discretization_equal_width(file_path):
    '''
    we first imagine that the optimized bins for each column is 10.
    
    Parameters:
    ===========
    file_path: the path of file which should be discretized. each column is digitized separately.
    
    Returns:
    ==========
    data
    
    '''
    data = read_data_from_PCA_output_file(file_path)
    _, cols = np.shape(data)
    
    for i in range(cols-1): # the last column is Person and is not processed
        selected_col = data[: , i]
        bins = np.linspace(np.amin(selected_col), np.amax(selected_col), 10)# devide the distance between min and max to 10 parts
        digitized = np.digitize(selected_col, bins)
        data[: , i] = digitized  

    return data.astype(int)

def discretization_equal_width_for_any_data(data):
    '''
    we first imagine that the optimized bins for each column is 10.
    
    Parameters:
    ===========
    file_path: the path of file which should be discretized. each column is digitized separately.
    
    Returns:
    ==========
    data
    
    '''
    _, cols = np.shape(data)
    
    for i in range(cols-1): # the last column is Person and is not processed
        selected_col = data[: , i]
        bins = np.linspace(np.amin(selected_col), np.amax(selected_col), 4)# devide the distance between min and max to 10 parts
        digitized = np.digitize(selected_col, bins)
        data[: , i] = digitized  
        print(data[:,i])

    return data.astype(int)

def shift_data(data):
    list_of_data = list(sorted(set(data)))
    for item in list_of_data:
        data[np.where(np.equal(data, item))] = list_of_data.index(item)
        
    #print(set(data))
    return data 

def digitize_Dr_Amirkhani(a, n):
    '''
    digitize a into n bins
    
    Parameters:
    ============
    a: ndarray
    n: # of bins
    
    '''
    mn = np.min(a)
    mx = np.max(a)
    
    step = (mx-mn)/n
    #print("step:" , step)
    b = np.zeros_like(a)
    for i in range(n):
        ind = np.where(np.logical_and(np.greater_equal(a,mn+i*step),np.less(a,mn+(i+1)*step)))
        #print("i: {} , ind:{}".format(i , ind))
        b[ind] = i
    #print(b)
    #the max item is not considered, so set it manually
    b[np.where(np.equal(a , mx))] = n-1
    
    if len(set(b)) != max(list(set(b))) + 1 : # in natural condition, len = max(set) + 1 (because the elemetns start in 0)
        #print("yesssss")
        b = shift_data(b)
        
    return b

def digitize_dataset(data_address, selected_bin, address_to_save , isSave = True):
    
    '''
    digitize a dataset based on selected_bin
    the source data file is an exported  PCA file
    
    Parameters:
    ===========
    data_address:
    selected_bin:
    address_to_save: 
    
    '''
    
    data = read_data_from_PCA_output_file(data_address)
    _ , cols = np.shape(data)
    
    for i in range(0,cols-1):# digitize each column seperately
        data[:,i] = digitize_Dr_Amirkhani(data[:,i], selected_bin)
   
    data = data.astype(int)
    
    if isSave:
        np.savetxt(address_to_save, data , delimiter=',' , fmt='%s')
        
    return data
    

def test_digitize_dataset_for_feature_enginnering_with_delta():
    
    '''
    just for digitizing 2 feature engineering methods: (no overlap and activity + delta)
    both train and test
    '''
    base_file_address_activity_and_delta = r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_activity_and_delta\{}\delta={}\PCA_n={}.csv"
    address_to_save_activity_and_delta = r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_activity_and_delta\{}\delta={}\digitize_bin_{}"#\PCA_n={}.csv"
  
    base_file_address_no_overlap = r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_no overlap\{}\delta={}\PCA_n={}.csv"
    address_to_save_no_overlap = r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_no overlap\{}\delta={}\digitize_bin_{}"#\PCA_n={}.csv"
  
    selected_bin = 5
    
    list_of_base_addresses = [base_file_address_activity_and_delta , base_file_address_no_overlap]
    list_of_save_addresses = [address_to_save_activity_and_delta , address_to_save_no_overlap]
    base = np.zeros(len(list_of_base_addresses) , dtype = object)
    save = np.zeros(len(list_of_save_addresses) , dtype = object)
          
    
    for delta in [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]:#
        for n in range(2,41):
              
            for b , s ,i in zip(list_of_base_addresses, list_of_save_addresses , range(len(list_of_base_addresses))) :
                base[i] = b.format('train', delta , n)
                save[i] = s.format('train', delta ,selected_bin )
            
                if not os.path.exists(save[i]):
                    os.makedirs(save[i])
                
                save[i] = save[i] + r"\PCA_n=" + str(n) + ".csv"
                digitize_dataset(data_address = base[i], selected_bin = selected_bin, address_to_save = save[i])

    
def test_digitize_dataset_based_on_activity():
    
    base_file_address = r"C:\pgmpy\separation of train and test\31_3\PCA on bag of sensor events_based on activity\train\PCA_n={}.csv"
    address_to_save = r"C:\pgmpy\separation of train and test\31_3\PCA on bag of sensor events_based on activity\train\digitize_bin_{}"#\PCA_n={}.csv"
  
    selected_bin = 200
    
    for n in range(2,41):
        
        base = base_file_address.format(n)
        save = address_to_save.format(selected_bin )
        
        
        if not os.path.exists(save):
            os.makedirs(save)
            
        save = save + r"\PCA_n=" + str(n) + ".csv"
        digitize_dataset(data_address = base, selected_bin = selected_bin, address_to_save = save)

    
def discretization_equal_frequency():
    pass


def create_PCA_for_different_bag_of_sensor_events_no_overlap():
    
    for delta in [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]:#15,30,45,60
        
        train_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_no overlap\train\delta=' + str(delta)
        test_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_no overlap\test\delta=' + str(delta)
        for directory in [train_directory_for_save , test_directory_for_save]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        train_directory_for_save = train_directory_for_save + '\\'
        test_directory_for_save = test_directory_for_save + '\\'
        
        train_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_no overlap_based on different deltas\train\delta_{}min.csv'.format(delta)
        test_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_no overlap_based on different deltas\test\delta_{}min.csv'.format(delta)


        PCA_data_generation_on_separated_train_and_test(file_address = train_file_address, base_address_to_save = train_directory_for_save, remove_date_and_time = False, remove_activity_column = False, test_file_address = test_file_address, base_address_of_test_file_to_save = test_directory_for_save)





def create_PCA_for_different_bag_of_sensor_events_based_on_activity_and_delta():
    
    for delta in [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]:#15,30,45,60
        
        train_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_activity_and_delta\train\delta=' + str(delta)
        test_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_activity_and_delta\test\delta=' + str(delta)
        for directory in [train_directory_for_save , test_directory_for_save]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        

        train_directory_for_save = train_directory_for_save + '\\'
        test_directory_for_save = test_directory_for_save + '\\'
        
        train_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\train\delta_{}min.csv'.format(delta)
        test_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_based_on_activity_and_no_overlap_delta\test\delta_{}min.csv'.format(delta)


        PCA_data_generation_on_separated_train_and_test(file_address = train_file_address, base_address_to_save = train_directory_for_save, remove_date_and_time = False, remove_activity_column = True, test_file_address = test_file_address, base_address_of_test_file_to_save = test_directory_for_save)


def create_PCA_for_bag_of_sensor_events_based_on_activities():

        
    train_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on bag of sensor events_based on activity\train'
    test_directory_for_save = r'C:\pgmpy\separation of train and test\31_3\PCA on bag of sensor events_based on activity\test'
    for directory in [train_directory_for_save , test_directory_for_save]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    train_directory_for_save = train_directory_for_save + '\\'
    test_directory_for_save = test_directory_for_save + '\\'
    
    train_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_based on activities\train\based_on_activities.csv'
    test_file_address = r'C:\pgmpy\separation of train and test\31_3\Bag of sensor events_based on activities\test\based_on_activities.csv'
    
    PCA_data_generation_on_separated_train_and_test(file_address = train_file_address, base_address_to_save = train_directory_for_save, remove_date_and_time = False, remove_activity_column = True, test_file_address = test_file_address, base_address_of_test_file_to_save = test_directory_for_save)

    

def test_discretization_on_different_PCA_data_files():  
    base_address = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events"
    base_save_address = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events_Digitized"
    
    for delta in [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]:
        for n in range(2,41):
            
            data = discretization_equal_width(base_address + r"\delta=" + str(delta) + r"\PCA_n=" + str(n)+ r".csv") 
            #if delta==15 and n==2:
            #   print(data)
                
            save_address = base_save_address + r"\delta=" + str(delta) 
            if not os.path.exists(save_address):
                os.makedirs(save_address)
            
            save_address = save_address + "\PCA_n=" + str(n)+ ".csv"  
            #print(save_address) 
            np.savetxt(save_address , data, delimiter=',' , fmt='%s')

                 
        
if __name__ == "__main__":
    
    #create_PCA_for_different_bag_of_sensor_events()
    #create_BN_model()
    #myData = np.genfromtxt(dest_file , dtype=object,delimiter = ',')#, names=False)
    #PCA_data_generation(dest_file)
    #print(read_data_from_file(dest_file, np.int, remove_date_and_time=True))
    #create_PCA_for_bag_of_sensor_events_based_on_activities()    
    #create_PCA_for_different_bag_of_sensor_events_based_on_activity_and_delta()
    #create_PCA_for_different_bag_of_sensor_events_no_overlap()
    #test_digitize_dataset_for_feature_enginnering_with_delta()
    a = [87, 1 , 2 , 0 , 13]
    print(digitize_Dr_Amirkhani(a, 10))
    #create_PCA_for_bag_of_sensor_events_based_on_activities()
    #create_PCA_for_different_bag_of_sensor_events()
    #print(shift_data(np.array([1,9])))
    #test_discretization_on_different_PCA_data_files()
    #f = r"\\localhost\E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events_Digitized\delta=15\alaki.csv"
    #f = r"E:/Lessons_tutorials/Behavioural user profile articles/Datasets/7 twor.2009/twor.2009/converted/pgmpy/PCA on Bag of sensor events_Digitized/delta=15/PCA_n=9.csv"

    #f = "C:/alaki.csv"
    #read_data_from_PCA_digitized_file(f)
    '''try:
        result = pd.read_csv(filepath_or_buffer = f , header = None)# , dtype = np.int32)
        print(result)
    except ValueError:
        print(ValueError)
       ''' 
    #read_data_from_PCA_digitized_file(f)