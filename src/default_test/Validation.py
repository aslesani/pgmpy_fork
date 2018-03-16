'''
Created on September 02, 2017

@author: Adele
'''

import numpy as np
import pandas as pd
from DimensionReductionandBNStructureLearning import create_BN_model
from DimensionReductionandBNStructureLearning import read_data_from_PCA_digitized_file
from DimensionReductionandBNStructureLearning import discretization_equal_width_for_any_data
from DimensionReductionandBNStructureLearning import digitize_Dr_Amirkhani, shift_data
from DimensionReductionandBNStructureLearning import read_data_from_PCA_output_file
from pgmpy.estimators import BdeuScore, K2Score, BicScore


from Abdollahi import bic

import matplotlib.pyplot as plt
import random

from sklearn import datasets
from sklearn.utils import shuffle

import cProfile


from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from pandas.core.frame import DataFrame
from builtins import int
from sklearn.metrics.classification import precision_score, recall_score,\
    accuracy_score
from numba.tests.npyufunc.test_ufunc import dtype


def convert_numpy_dataset_to_pandas(data):
    '''
    get a numpy dataset and convert it to pandas dataframe
    Parameters:
    ==========
    data: an ndarray numpy 
    
    Retrun:
    =======
    Dataframe
    '''
    
    _ , cols = data.shape
    column_names = ['c' + str(i) for i in range(cols-1)]
    column_names.append('Person')
    pd_data_set = pd.DataFrame(data , columns=column_names)
    return pd_data_set
            

def partition_data(data, train_ratio, validation_ratio, test_ratio):
    '''
    partition data to train, validation and test sets according to proportion
   
    Parameters:
    ===========
    data: numpy ndarray
    train_ratio, validation_ratio, test_ratio: the % of each set (a number less than 100)
    
    '''
    
    np.random.shuffle(data)
    data_length = len(data)
    print("data_length: {}".format(data_length))
    
    validation_length, test_length = int(data_length * validation_ratio /100) , int(data_length * test_ratio /100)
    train_length = data_length - validation_length - test_length
    print("train_length: {}, validation_length:{} , test_length:{}".format(train_length, validation_length, test_length))
    
    train_set = data[0:train_length]
    validation_set = data[train_length: (train_length + validation_length)]
    test_set = data[(train_length + validation_length) : data_length]
    
    return(train_set, validation_set, test_set)

    
    
def kfoldcrossvalidationForBNModelUsingNumpy(k , data, target_column_name, scoring='f1_micro' ):  
    '''
    get a model and test it using k-fold cross validation
    Ù�Ø¹Ù„Ø§ ÛŒÚ© Ø§ÛŒØ±Ø§Ø¯ Ø¯Ø§Ø±Ø¯ Ú©Ù‡ Ù…Ù† Ø¨Ø±Ø§ÛŒ Ø±Ø§Ø­ØªÛŒ Ú©Ø§Ø± Ù‡Ù…Ù‡ Ø¯Ø³ØªÙ‡ Ù‡Ø§ Ø±Ø§ ÛŒÚ© Ø§Ù†Ø¯Ø§Ø²Ù‡ Ú¯Ø±Ù�ØªÙ…. Ù…Ø§Ú©Ø³ÛŒÙ…Ù… Ø¨Ù‡ ØªØ¹Ø¯Ø§Ø¯ 
    k-1
    Ø¯Ø§Ø¯Ù‡ Ù…Ù…Ú©Ù† Ø§Ø³Øª Ø§ØµÙ„Ø§ Ø¯Ø± Ø§Ø±Ø²ÛŒØ§Ø¨ÛŒ ÙˆØ§Ø±Ø¯ Ù†Ø´ÙˆÙ†Ø¯. Ø¨Ø¹Ø¯Ø§ Ø¯Ø±Ø³Øª Ù…ÛŒ Ú©Ù†Ù….
    
    Ù�Ø¹Ù„Ø§ Ù�Ø±Ø¶ Ú©Ø±Ø¯Ù‡ Ø§Ù… Ø³ØªÙˆÙ† Ø¢Ø®Ø± ØªØ§Ø±Ú¯Øª Ø§Ø³Øª
    
    Parameters:
    -------
    k: the value of k in k-fold
    data: a pandaFrame object the estimator_model would be tested on 
          (we imagine the last column is the target)
          
    scoring: the scoring method , default value:'f1_micro'
    
    Returns
    -------
    scores: a vector of scores, the length of the vector is k.
    
    ''' 
    #np.random.shuffle(data)
    data_length , data_features = np.shape(data)
    #print(data_features)
    #print("data_length: {}".format(data_length))
    part_length = int(data_length / k)
    
    p = np.zeros((k, part_length, data_features), dtype= np.int)
    partition = pd.DataFrame(p , columns=['1','2','3','4'])
    
    index = 0
    
    #print("type(data):{}, type(partition){}".format(type(data), type(partition)))
    # the last k is partitioned manually, because maybe hte data_length was not devided on k
    for i in range(0, k): 
        end = (i+1) * part_length
        print("index:{}, end: {}".format(index, end))
        partition[i] = data[index:end]
        index = end
    
    scores = np.zeros(k)
    
    for i in range(0,k):
        test_set = partition[i]
        train_set =  np.delete(partition, i, 0)  
        #convert 3d array to 2d array, C means read elements like the order of C matrixes
        train_set = np.reshape(train_set, ((k-1)*part_length, data_features), order="C")
        estimator , _ = create_BN_model(train_set)
        #print("here (k={})".format(i+1))# , estimator.get_parameters))
        
        test_set_pd = pd.DataFrame(test_set, columns=['1','2','3','4'])
        scores[i] = pgm_test(estimator, test_set_pd, target_column_name)
        #print(np.shape(train_set))
        #print(train_set)
        #print(np.shape(test_set))
        #print(test_set)

    return scores    
    

def kfoldcrossvalidationForBNModel_UsingPanda(k , data, target_column_name, scoring='f1_micro' ):  
    '''
    get a model and test it using k-fold cross validation
    Ù�Ø¹Ù„Ø§ ÛŒÚ© Ø§ÛŒØ±Ø§Ø¯ Ø¯Ø§Ø±Ø¯ Ú©Ù‡ Ù…Ù† Ø¨Ø±Ø§ÛŒ Ø±Ø§Ø­ØªÛŒ Ú©Ø§Ø± Ù‡Ù…Ù‡ Ø¯Ø³ØªÙ‡ Ù‡Ø§ Ø±Ø§ ÛŒÚ© Ø§Ù†Ø¯Ø§Ø²Ù‡ Ú¯Ø±Ù�ØªÙ…. Ù…Ø§Ú©Ø³ÛŒÙ…Ù… Ø¨Ù‡ ØªØ¹Ø¯Ø§Ø¯ 
    k-1
    Ø¯Ø§Ø¯Ù‡ Ù…Ù…Ú©Ù† Ø§Ø³Øª Ø§ØµÙ„Ø§ Ø¯Ø± Ø§Ø±Ø²ÛŒØ§Ø¨ÛŒ ÙˆØ§Ø±Ø¯ Ù†Ø´ÙˆÙ†Ø¯. Ø¨Ø¹Ø¯Ø§ Ø¯Ø±Ø³Øª Ù…ÛŒ Ú©Ù†Ù….
    
    Ù�Ø¹Ù„Ø§ Ù�Ø±Ø¶ Ú©Ø±Ø¯Ù‡ Ø§Ù… Ø³ØªÙˆÙ† Ø¢Ø®Ø± ØªØ§Ø±Ú¯Øª Ø§Ø³Øª
    
    Parameters:
    -------
    k: the value of k in k-fold
    data: a pandaFrame object the estimator_model would be tested on 
          (we imagine the last column is the target)
          
    scoring: the scoring method , default value:'f1_micro'
    
    Returns
    -------
    scores: a vector of scores, the length of the vector is k.
    
    ''' 
    #np.random.shuffle(data)
    data = shuffle(data)
    data_length , data_features = np.shape(data)
    #print(data_features)
    #print("data_length: {}".format(data_length))
    part_length = int(data_length / k)
    
    partition = np.zeros(k , dtype= pd.DataFrame)
    #partition = pd.DataFrame(p , columns=['1','2','3','4'])
    
    index = 0
    
    #print("type(data):{}, type(partition){}".format(type(data), type(partition)))
    # the last k is partitioned manually, because maybe hte data_length was not devided on k
    for i in range(0, k): 
        end = (i+1) * part_length
        #print("index:{}, end: {}".format(index, end))
        partition[i] = data[index:end]
        index = end
    
    scores = np.zeros(k)
    
    for i in range(0,k):
        #if i != 1:
        #    continue
        
        test_set = partition[i]
        train_set =  np.delete(partition, i, 0)
        
        #print("train_set.shape:{}".format(train_set.shape))
        #create panda dataframe from nd array
        pd_train_set = train_set[0]
        #print("train_set[0]:{}".format(train_set[0]))
        for j in range(1, k-1):
            pd_train_set = pd_train_set.append(train_set[j]) 
        #end    
            
        #print(type(pd_train_set))
        #print("pd_train_set:{}".format(pd_train_set))
        
        
        #convert 3d array to 2d array, C means read elements like the order of C matrixes
        #train_set = np.reshape(train_set, ((k-1)*part_length, data_features), order="C")
        estimator , learning_time = create_BN_model(pd_train_set)
        #print("here (k={})".format(i+1))# , estimator.get_parameters))
        
        #test_set_pd = pd.DataFrame(test_set, columns=['1','2','3','4'])
    
        scores[i] = pgm_test(estimator, test_set, target_column_name)
        #print(np.shape(train_set))
        #print(train_set)
        #print(np.shape(test_set))
        #print(test_set)

    print("scores:{}".format(scores))
    return scores.mean()# , learning_time  

def kfoldcrossvalidation_for_abd_function(k , data, data_column_names, target_column_name ):  
    '''
    
    the function is used for creating and testing the models. 
    models are created by Abd function
    
    Parameters:
    -------
    k: the value of k in k-fold
    data: numpy ndarray
          
    #scoring: the scoring method , default value:'f1_micro'
    
    Returns
    -------
    scores: a vector of scores, the length of the vector is k.
    
    ''' 
    data_length , data_features = np.shape(data)
    #print(data_features)
    #print("data_length: {}".format(data_length))
    part_length = int(data_length / k)
    
    #p = np.zeros((k, part_length, data_features), dtype= np.int)
    #print(np.shape(p))
    partition = np.zeros(k , dtype = pd.DataFrame)#pd.DataFrame(p , columns=['1','2','3','4'])
    
    index = 0
    
    # the last k is partitioned manually, because maybe the data_length was not devided on k
    for i in range(0, k): 
        end = (i+1) * part_length
        #print("index:{}, end: {}".format(index, end))
        partition[i] = pd.DataFrame(data[index:end] , dtype = np.int, columns = data_column_names)
        index = end
    
    scores = np.zeros(k , dtype = np.object)
    
    i = 0
    while i < k:
        validation_set = partition[i]
        train_set =  np.delete(partition, i, 0)  
        train_set = pd.concat(train_set)#[partition(p) for p in len(train_set)]
        #print(type(test_set))
        #validation_set = pd.DataFrame(test_set , columns=data_column_names)    
        resultlist = validation_set[target_column_name].values
        test_final = validation_set.drop(target_column_name, axis=1, inplace=False)
        
        #print("train_set:\n" , train_set)
        #print("test_final:\n" , test_final)
        
        try:
            _ , scores[i], _ , _ = bic(train = train_set,test = test_final, scoring_function = BicScore , resultlist = resultlist)
            #i = i + 1
        except ValueError:
            #print("try again for k ={} in k-fold cross validation".format(i))
            print("exception")
            scores[i] = 0
            
        i = i + 1   
        
    return scores    
    

def pgm_test(estimator, test_set, target_column_name):
    '''
    test the pgm model. the model is trained by a train_set. 
    
    Parameters:
    =========== 
    estimator: the pgm model that should be tested 
    
    test_set: clear ;) the type of it is panda dataframe
        
    target_column_name: the name of the column that should be predicted
    
    Return:
    =======
    score
    '''
    
    print("hello")
    y_true = test_set[target_column_name].values#, y_predicted = np.zeros(1, number_of_tests)
    #y_true = test_set.iloc[:,target_column_name]
    #print(y_true)
    #y_true = pd.DataFrame.as_matrix(y_true)
    #print("@@@@@@@@@@@@@@@@@@@@@")
    #print(type(y_true.values))
    
    test_set = test_set.drop(target_column_name, axis=1, inplace=False)
    print(test_set)

    #y_predicted = np.zeros(shape=y_true.shape , dtype = int)
    print(type(estimator))
    y_predicted = estimator.predict(test_set)

    
    '''
    rows , _ = test_set.shape
    
    for i in range (0 ,rows):
        #print("=================\ni:{}\n=================".format(i))
        #print("test data:\n {}".format(test_set.iloc[[i]]))
        #print("i : " , i)
        print(test_set.iloc[[i]])
        a = estimator.predict(test_set.iloc[[i]])
        #y_predicted.iloc[[i]] = a
        y_predicted[i] = a.iloc[0].values
        #print(y_true[i] ,'\n', a , '\n' , y_predicted[i])

    '''
    print("y_true:\n{}\ny_predicted:\n{}".format( y_true, y_predicted))
    #score = f1_score(y_true, y_predicted, average='micro') 
    score = calculate_different_metrics(y_true, y_predicted)
    return score
    
def calculate_different_metrics(y_true , y_predicted):
    f1_score_micro = f1_score(y_true, y_predicted, average='micro') 
    f1_score_macro = f1_score(y_true, y_predicted, average='macro') 
    precision = precision_score(y_true, y_predicted, average='micro') 
    recall = recall_score(y_true, y_predicted, average='micro')
    accuracy = accuracy_score(y_true, y_predicted)  
    
    scores = {'f1_score_micro': f1_score_micro, 
              'f1_score_macro': f1_score_macro,
              'precision' : precision,
              'recall' : recall,
              'accuracy' : accuracy
              }
    
    return scores

                
def kfoldcrossvalidationUsingpgmpy(k , data, target, estimator_model, scoring='f1_micro' ):  
    scores = cross_val_score(estimator_model, data, target, cv=10 , scoring=scoring) ####10-fold cross validation 
    return scores
    
def create_empty_2d_array(row , col):
    
    #1d array 
    #arr = np.arange(100)
    arr = np.arange(20).reshape((row , col))#np.zeros(shape=(row,col))
    print(arr)
    return arr
    
def test_pgm_test():
    data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})#, columns=[1,2,3])
    
    print(data)
    
    model = BayesianModel([('A', 'C'), ('B', 'C')])
    model.fit(data, estimator=BayesianEstimator, prior_type="K2")#MaximumLikelihoodEstimator)

    test_set = pd.DataFrame(data={'A': [0, 0], 'B': [0, 1], 'C': [0, 1]})
    
    print(pgm_test(model, test_set, target_column_name = 'C').mean())
    

    
def prepare_data_to_create_model_and_test(delta):
    '''
    read data from address_file and add columns names to pass to kfoldcrossvalidation method
    
    '''
    base_address = r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events_Digitized\delta=" + str(delta) 
    scores = np.zeros(39)
    learning_time = np.zeros(39)
    for n in range(2,41):
        data = read_data_from_PCA_digitized_file(base_address + "\PCA_n=" + str(n) +".csv")
        sc , learning_time[n-2] = kfoldcrossvalidationForBNModel_UsingPanda(10, data, target_column_name = "Person", scoring = "f1_micro")
        scores[n-2] = sc.mean()
        
    print(list(range(2,41)))
    print(scores)
    print(learning_time)

def plot_results(x_values , y_values, x_label, y_label):
    '''
    plot the figure
    
    Parameters:
    ===========
    x_values: the list of x values
    y_values: the list of y values
    x_label:
    y_label:
    
    '''
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(x_values, y_values)#, linewidth=1)
    #plt.plot(x_values ,  [1,1,1,1])
    #plt.axis('tight')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    ###############################################################################
    # Prediction
    '''
    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)
    
    #Parameters of pipelines can be set using â€˜__â€™ separated parameter names:
    
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs))
    
    plt.axvline(n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    '''
    plt.show()

def test_with_iris_dataset():
    
    iris = datasets.load_iris()
    data = iris.data#[:, :2]  # we only take the first two features. We could
                        # avoid this ugly slicing by using a two-dim dataset
    target = iris.target
    
    target.shape = (150,1)
    
    pd_data = pd.DataFrame(np.concatenate((data, target), axis=1) , columns = ['c1' , 'c2' , 'c3' , 'c4' , 'target'])
    
    #print(pd_data.columns)
    pd_data.c1 = pd_data.c1.astype(np.int64)
    pd_data.c2 = pd_data.c2.astype(np.int64)
    pd_data.c3 = pd_data.c3.astype(np.int64)
    pd_data.c4 = pd_data.c4.astype(np.int64)
    pd_data.target = pd_data.target.astype(np.int64)

    #print(pd_data)
    #return pd_data
    print(kfoldcrossvalidationForBNModel_UsingPanda(10, pd_data, target_column_name = "target", scoring = "f1_micro"))

   
def kaggle_dataset():
    data = pd.read_csv("kaggle.csv")
    print(data)

    data2 = data[["PassengerId" , "Survived", "Sex", "Pclass"]]
    #data2 = data[["Survived", "Sex", "Pclass"]].replace(["female", "male"], [0, 1]).replace({"Pclass": {3: 0}})
    
    intrain = np.random.rand(len(data2)) < 0.8
    
    dtrain = data2[intrain]
    dtest = data2[~intrain]
    
    ##print(len(dtrain), len(dtest))
    
    from pgmpy.models import BayesianModel
    titanic = BayesianModel()
    titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived"), ("Pclass", "PassengerId")])
    titanic.fit(dtrain)
    for cpd in titanic.get_cpds():
        print(cpd)
    
    
    print(titanic.predict(dtest[["Survived", "Sex", "Pclass"]]))
    pgm_test(titanic, dtest, "PassengerId")
   
    
    
def disretization_Dr_Amirkhani(data):
    '''
    data is ndarray
    
    Return:
    ======
    data: panda dataframe
    '''
    column_names = []
    for i in range(data.shape[1]-1):# skip last columns (target)
        data[:,i] = digitize_Dr_Amirkhani(data[:,i], 4)
        if len(set(data[:,i])) != max(list(set(data[:, i]))) + 1 : # in natural condition, len = max(set) + 1 (because the elemetns start in 0)
            data[:,i] = shift_data(data[:,i])
        
        column_names.append("c" + str(i))
    data = data.astype(int)
    column_names.append("Person")
    data = pd.DataFrame( data , columns = column_names)
    
    return data

def iris_prepare_for_discritization():

    iris = datasets.load_iris()
    data = iris.data

    for i in range(data.shape[1]):
        data[:,i] = digitize_Dr_Amirkhani(data[:,i], 4)
        #print(data[:,1])
    
    target = iris.target
    target.shape = (150,1)
    #print(data.shape, target.shape )
    data = data.astype(int)
    target = target.astype(int)
    data = pd.DataFrame( np.concatenate((data, iris.target), axis=1) , columns = ['c1' , 'c2' , 'c3' , 'c4' , 'Person'])
    #print(data)
    return data

    
def iris_dicretization():

    # irad dare in code!!!!!!
    #ba iris test mikonam. value ha ra az 0 gereftam (min 0)
    
    iris = datasets.load_iris()
    data = iris.data#[:, :2]  # we only take the first two features. We could
                        # avoid this ugly slicing by using a two-dim dataset
    target = iris.target
    
    target.shape = (150,1)
    
    digitized_data = discretization_equal_width_for_any_data(np.concatenate((data, target), axis=1))
    
    for i in range(0,5):
        #print(set(digitized_data[:, i]))
        digits_set = list(set(digitized_data[:, i]))
        if len(digits_set) < 4:
            for row in range(0,150):
                digitized_data[row , i] = digits_set.index(digitized_data[row , i])
        elif 0 not in digits_set:# the digits_set have all number, but the index is not started from 0
            print("0 nit in digits_set")
            digitized_data[:, i] -=1
            
        print(set(digitized_data[:, i]))   
    
    pd_data = pd.DataFrame( digitized_data , columns = ['c1' , 'c2' , 'c3' , 'c4' , 'Person'])

    return pd_data
       
    
    #print(pd_data.columns)
    #pd_data.c1 = pd_data.c1.astype(np.int64)
    #pd_data.c2 = pd_data.c2.astype(np.int64)
    #pd_data.c3 = pd_data.c3.astype(np.int64)
    #pd_data.c4 = pd_data.c4.astype(np.int64)
    #pd_data.target = pd_data.target.astype(np.int64)

    #print(pd_data)
    #return pd_data

    
def select_hyperparameters():
    delta = [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]
    delta_length = len(delta)-1
    details_of_each_repeat = []
    max_validation_score = {'f1_score_micro': 0, 
                              'f1_score_macro': 0,
                              'f1_score_binary': 0,
                              'precision' : 0,
                              'recall' : 0,
                              'accuracy' : 0
                              }
    scores_abd = max_validation_score.copy()
    the_best_model = 0
    best_model_pd_test_set = 0
    best_delta = 0
    best_n = 0
          
    feature_engineering_names = ["delta_no overlap" , "activity" , "activity and delta"]      
    for repeat in range(2,3): # repaet the process of selecting hyperparameters
        print("repeat: {}".format(repeat))
        selected_delta = 90#delta[random.randint(1,delta_length)]
        selected_n = 8#random.randint(2,10)#41)# n is # of features in PCA
        print("selected_delta:{} , selected_n:{}".format(selected_delta,selected_n))
        #print("n = " , selected_n)
    
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on bag of sensor events_based on activity\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on Bag of sensor events_no overlap\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on Bag of sensor events\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")

        based_on_delta_no_overlap = r"C:\pgmpy\PCA on Bag of sensor events_no overlap\delta={}\PCA_n={}.csv".format(selected_delta , selected_n)
        based_on_activity = r"C:\pgmpy\PCA on bag of sensor events_based on activity\PCA_n={}.csv".format(selected_n)
        based_on_activity_and_delta = r"C:\pgmpy\PCA on Bag of sensor events_activity_and_delta\delta={}\PCA_n={}.csv".format(selected_delta , selected_n)
        #data = read_data_from_PCA_digitized_file(r"C:\pgmpy\Bag of sensor events_no overlap_based on different deltas\bag_of_sensor_events_no_overlap_delta_" + str(selected_delta) + "min.csv")
        #data = data[0:5000, :]
        #print(data.shape)
        
        addresses = [based_on_activity_and_delta]#based_on_delta_no_overlap]# , based_on_activity , based_on_activity_and_delta]
        for data_address , feature_engineering_name in zip(addresses , feature_engineering_names):
            
            data = read_data_from_PCA_output_file(data_address)
            
            for i in range(0,selected_n):# digitize each column seperately
                
                feature_set_length = len(set(data[:,i]))
                #print("column number: {}, number of states:{}".format(i, feature_set_length))
                selected_bin = 10#random.randint(2, 1000)#feature_set_length)
                #print("feature_set_length:{}, selected_bin:{}".format(feature_set_length, selected_bin))
                data[:,i] = digitize_Dr_Amirkhani(data[:,i], selected_bin)
            
            data = data.astype(int)
            
            #np.savetxt(r"C:\pgmpy\PCA on Bag of sensor events\digitized\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + "_Bin=" + str(selected_bin) +".csv", 
            #       data, delimiter=',' , fmt='%s')
    
            
            #yadet bashe shuffle ro comment kardi
            train_set, validation_set, test_set = partition_data(data, train_ratio = 80, validation_ratio = 10, test_ratio = 10)
            column_names = ['c' + str(i) for i in range(selected_n)]
            column_names.append('Person')
            pd_train_set = pd.DataFrame(train_set , columns=column_names)
            pd_validation_set = pd.DataFrame(validation_set , columns=column_names)
            
            #pd_validation_set = pd_validation_set[0:100]
            resultlist = pd_validation_set['Person'].values
            test = pd_validation_set.drop('Person', axis=1, inplace=False)
            
            for sc in [BicScore]:# , BdeuScore , K2Score]:
                #model , scores_abd , learning_time_abd , testing_time_abd = bic(train = pd_train_set, test = test, scoring_function= sc, name = "model", folder = "Abdollahi", resultlist = resultlist, address = "C:\\")
                
                try:
                    model , scores_abd , learning_time_abd , testing_time_abd = bic(train = pd_train_set, test = test, scoring_function= sc, resultlist = resultlist)
                except ValueError:
                    print("pd_train_set:\n " ,pd_train_set)
                    print("test:\n" , test)
                #print(sc)
                #print("Abd execution\n========================\n")
                print( feature_engineering_name , scores_abd)# , "\nlearning time: " , learning_time_abd , 
                    #"\ntesting time: " , testing_time_abd)      
               
                #estimator , learning_time_adi = create_BN_model(pd_train_set)
               
                #validation_score_adi = pgm_test(estimator, pd_train_set, 'Person')
            
                '''
                pd_test_set = pd.DataFrame(test_set , columns=column_names)
                resultlist = pd_test_set['Person'].values
                test = pd_test_set.drop('Person', axis=1, inplace=False)
                predicted = model.predict(test).values.ravel()
                model_test_set_score = calculate_different_metrics(resultlist, predicted)
                print("test set:")
                print(model_test_set_score)
                '''
              
        #print(scores_abd['f1_score_micro'])
        #print( max_validation_score['f1_score_micro'])
        if scores_abd['f1_score_micro'] > max_validation_score['f1_score_micro']:
            max_validation_score = scores_abd
            the_best_model = model
            #best_model_pd_test_set = pd_test_set
            best_delta = selected_delta
            best_n = selected_n
    
        #print("validation_score: " , validation_score)
        details_of_each_repeat.append([selected_delta, selected_n, learning_time_abd , testing_time_abd , scores_abd , model])
        
    '''
    #test the best model with test set
    resultlist = best_model_pd_test_set['Person'].values
    test = best_model_pd_test_set.drop('Person', axis=1, inplace=False)
    predicted = the_best_model.predict(test).values.ravel()
    best_model_test_set_score = calculate_different_metrics(resultlist, predicted)
    '''
    print("\n=======Best Parameters:=======\n")
    print("best delta= " , best_delta , " best n: " , best_n)
    print("best validation score:" , max_validation_score )#, "\n best test score: " , best_model_test_set_score)
        
    
    return (max_validation_score , the_best_model , best_model_pd_test_set, best_delta , best_n , details_of_each_repeat)
    
    return (0,0,0,0,0,0)


def select_hyperparameters_with_hillclimb_strategy_and_split_train_test():
    
    delta = [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]
    delta_length = len(delta)-1
    details_of_each_repeat = []
    max_validation_score = {'f1_score_micro': 0.0, 
                              'f1_score_macro': 0.0,
                              'f1_score_binary': 0.0,
                              'precision' : 0.0,
                              'recall' : 0.0,
                              'accuracy' : 0.0
                              }
    the_best_model = 0
    best_model_pd_test_set = 0
    best_delta = 0
    best_n = 0
          
    feature_engineering_names = ["delta_no overlap" , "activity" , "activity and delta"]      
    for repeat in range(2,41): # repaet the process of selecting hyperparameters
        #print("repeat: {}".format(repeat))
        selected_delta = delta[random.randint(1,delta_length)]
        selected_n = random.randint(2,10)#41)# n is # of features in PCA
        print("selected_delta:{} , selected_n:{}".format(selected_delta,selected_n))
        #print("n = " , selected_n)
    
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on bag of sensor events_based on activity\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on Bag of sensor events_no overlap\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")
        #data = read_data_from_PCA_output_file(r"C:\pgmpy\PCA on Bag of sensor events\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + ".csv")#r"C:\f5_0_10.csv")

        based_on_delta_no_overlap = r"C:\pgmpy\PCA on Bag of sensor events_no overlap\delta={}\PCA_n={}.csv".format(selected_delta , selected_n)
        based_on_activity = r"C:\pgmpy\PCA on bag of sensor events_based on activity\PCA_n={}.csv".format(selected_n)
        based_on_activity_and_delta = r"C:\pgmpy\PCA on Bag of sensor events_activity_and_delta\delta={}\PCA_n={}.csv".format(selected_delta , selected_n)
        #data = read_data_from_PCA_digitized_file(r"C:\pgmpy\Bag of sensor events_no overlap_based on different deltas\bag_of_sensor_events_no_overlap_delta_" + str(selected_delta) + "min.csv")
        #data = data[0:5000, :]
        #print(data.shape)
        
        addresses = [based_on_activity_and_delta]#based_on_delta_no_overlap]# , based_on_activity , based_on_activity_and_delta]
        for data_address , feature_engineering_name in zip(addresses , feature_engineering_names):
            
            data = read_data_from_PCA_output_file(data_address)
            
            for i in range(0,selected_n):# digitize each column seperately
                
                feature_set_length = len(set(data[:,i]))
                #print("column number: {}, number of states:{}".format(i, feature_set_length))
                selected_bin = 10#random.randint(2, 1000)#feature_set_length)
                #print("feature_set_length:{}, selected_bin:{}".format(feature_set_length, selected_bin))
                data[:,i] = digitize_Dr_Amirkhani(data[:,i], selected_bin)
            
            data = data.astype(int)
            
            #np.savetxt(r"C:\pgmpy\PCA on Bag of sensor events\digitized\delta=" + str(selected_delta) + "\PCA_n=" + str(selected_n) + "_Bin=" + str(selected_bin) +".csv", 
            #       data, delimiter=',' , fmt='%s')
    
            
            #yadet bashe shuffle ro comment kardi
            train_set, validation_set, test_set = partition_data(data, train_ratio = 80, validation_ratio = 10, test_ratio = 10)
            column_names = ['c' + str(i) for i in range(selected_n)]
            column_names.append('Person')
            pd_train_set = pd.DataFrame(train_set , columns=column_names)
            pd_validation_set = pd.DataFrame(validation_set , columns=column_names)
            
            #pd_validation_set = pd_validation_set[0:100]
            resultlist = pd_validation_set['Person'].values
            test = pd_validation_set.drop('Person', axis=1, inplace=False)
            
            for sc in [BicScore]:# , BdeuScore , K2Score]:
                model , scores_abd , learning_time_abd , testing_time_abd = bic(train = pd_train_set, test = test, scoring_function= sc, name = "model", folder = "Abdollahi", resultlist = resultlist, address = "C:\\")
        
                #print(sc)
                #print("Abd execution\n========================\n")
                print( feature_engineering_name , scores_abd)# , "\nlearning time: " , learning_time_abd , 
                    #"\ntesting time: " , testing_time_abd)      
               
                #estimator , learning_time_adi = create_BN_model(pd_train_set)
               
                #validation_score_adi = pgm_test(estimator, pd_train_set, 'Person')
            
                '''
                pd_test_set = pd.DataFrame(test_set , columns=column_names)
                resultlist = pd_test_set['Person'].values
                test = pd_test_set.drop('Person', axis=1, inplace=False)
                predicted = model.predict(test).values.ravel()
                model_test_set_score = calculate_different_metrics(resultlist, predicted)
                print("test set:")
                print(model_test_set_score)
                '''
              
        #print(scores_abd['f1_score_micro'])
        #print( max_validation_score['f1_score_micro'])
        if scores_abd['f1_score_micro'] > max_validation_score['f1_score_micro']:
            max_validation_score = scores_abd
            the_best_model = model
            #best_model_pd_test_set = pd_test_set
            best_delta = selected_delta
            best_n = selected_n
    
        #print("validation_score: " , validation_score)
        details_of_each_repeat.append([selected_delta, selected_n, learning_time_abd , testing_time_abd , scores_abd , model])
        
    '''
    #test the best model with test set
    resultlist = best_model_pd_test_set['Person'].values
    test = best_model_pd_test_set.drop('Person', axis=1, inplace=False)
    predicted = the_best_model.predict(test).values.ravel()
    best_model_test_set_score = calculate_different_metrics(resultlist, predicted)
    '''
    print("=======Best Parameters:=======\n")
    print("best delta= " , best_delta , " best n: " , best_n)
    print("best validation score:" , max_validation_score )#, "\n best test score: " , best_model_test_set_score)
        
    
    return (max_validation_score , the_best_model , best_model_pd_test_set, best_delta , best_n , details_of_each_repeat)
    
    return (0,0,0,0,0,0)

def profiling():
    
    cProfile.run('re.compile("foo|bar")')
    
def BN_for_discritized_data():
    
    '''data = read_data_from_PCA_output_file(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events\delta=150\PCA_n=29.csv")
    
    data = disretization_Dr_Amirkhani(data)
    data.to_csv(r"D:\data.csv", sep = ',')
    '''
    #score = kfoldcrossvalidationForBNModel_UsingPanda(10, data, target_column_name = "Person", scoring = "f1_micro")
    #print("score:{}".format(score))
    
    #a = pd.read_csv(r"D:\data.csv")
    #print(a)
    data = read_data_from_PCA_digitized_file(r'C:\Users\Cloud\PCA on Bag of sensor events_Digitized\delta=15\PCA_n=6.csv')#(r"D:\data.csv")
    mul = range(1,13)
    sample_sizes = np.zeros(len(mul))
    learning_times = np.zeros(len(mul))
    for i in mul:
        step = i * 10000
        sample_sizes[i-1] = step
        print(i)
        pd_data = pd.DataFrame(data[0:step,:] , columns=['c0','c1','c2','c3','c4','c5','Person'])#'c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26','c27','c28','Person'])
        _ , learning_time = create_BN_model(pd_data)
        learning_times[i-1] = float("{0:.2f}".format(learning_time))
        
    print(sample_sizes , learning_times)
    plot_results(sample_sizes, learning_times, "sample_sizes", "learning_times")
    
def create_model_for_different_sample_size():
    data = read_data_from_PCA_output_file(r"E:\Lessons_tutorials\Behavioural user profile articles\Datasets\7 twor.2009\twor.2009\converted\pgmpy\PCA on Bag of sensor events\delta=100\PCA_n=6.csv")#("D:\data.csv")#
    data = disretization_Dr_Amirkhani(data)
    #learning_time = np.zeros_like(range(101), dtype = np.int)
    for i in range(2 , 138):#(1,101):
        train_set = data.iloc[0:i*500, :]
        print("i:{}".format(i))
        #pr = cProfile.Profile()
        #pr.enable()
        #print(train_set.shape)
    #_ , learning_time[i] = create_BN_model(train_set)
        #pr = cProfile.Profile()
        #pr.enable()
    
        create_BN_model(train_set)
        
        #pr.disable()
        #pr.print_stats(sort='time')
    
       
        #pr.disable()
        #pr.print_stats(sort='time')
        #print("i={}, learning_time:{}".format(i , learning_time[i]))
        
    #plot_results(range[101], y_values = learning_time, x_label = "samples(i*500)", y_label = "total learning time")
def test_create_BN_model(preffered_file , delta , n):
    data = read_data_from_PCA_digitized_file(preffered_file)
    
    _ , cols = data.shape
    column_names = []

    for names in range(1, cols):
        column_names.append('c' + str(names))
    column_names.append("Person")  

    pd_data = pd.DataFrame(data[:,:] , columns= column_names , dtype = np.int)#'c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26','c27','c28','Person'])

    _ , learning_time = create_BN_model(pd_data)
        
    print("learning time for delta = " , delta , " and n=" , n , ": " , learning_time)
    return learning_time
    
def test_create_BN_model_for_different_feature_numbers():
    dest = r'C:\dataset\casas7_dataset\delta=15\PCA_n='#40.csv'
    
    my_range = range(2,41)
    min_my_range = my_range[0]
    feature_numbers = np.zeros(len(my_range))
    learning_times = np.zeros(len(my_range))
    

    for num_of_features in my_range:
        new_dest = dest + str(num_of_features) + ".csv"
        #test_create_BN_model(new_dest, delta = 15, n = num_of_features)
    
        feature_numbers[num_of_features-min_my_range] = num_of_features
        lt = test_create_BN_model(new_dest, delta = 15, n = num_of_features)
        learning_times[num_of_features -min_my_range] = float("{0:.2f}".format(lt))    
        print("#features: " , num_of_features ,"learning time: " ,  lt)
        
    plot_results(feature_numbers, learning_times, "#features", "learning_time")
    
def the_best_validation_strategy(data, data_column_names, target_column_name , k = 10):
    '''
    a combination of split data and k-fold cross validation
    Imagine the final  test set is separated and the final train set is available. 
    Our validation approach split the data to validation and test set (90% and 10%)
    and then apply 10-fold cross validation on validation set.
    
    Parameters:
    =========== 
    data: numpy ndarray
    
    '''
    
    _ , validation_set , test_set =  partition_data(data, train_ratio = 0, validation_ratio = 90, test_ratio = 10)
    
    final_scores = kfoldcrossvalidation_for_abd_function(k = k, data = validation_set, data_column_names = data_column_names, target_column_name = target_column_name)
    final_f1_scores_micro_avg = 0
    for i in range (0,k):
        final_f1_scores_micro_avg = final_f1_scores_micro_avg + final_scores[i]['f1_score_micro'] 
        
    print("validation scores:" , final_scores , "f1 score average:" , final_f1_scores_micro_avg)
    
    pd_test_set = convert_numpy_dataset_to_pandas(test_set)
    pd_validation_set = convert_numpy_dataset_to_pandas(validation_set)
    resultlist = pd_test_set[target_column_name].values
    test_final = pd_test_set.drop(target_column_name, axis=1, inplace=False)

    
    _ , test_set_score, _ , _ = bic(train = pd_validation_set,test = test_final, scoring_function = BicScore , resultlist = resultlist)
    print("test score:" , test_set_score)
        
        
    

def test_the_best_validation_strategy():
    
    #data_address = r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_no overlap\train\delta=1000\PCA_n=10.csv"
    data_address = r"C:\f5_0_10.csv"#r"C:\pgmpy\separation of train and test\31_3\PCA on Bag of sensor events_activity_and_delta\train\delta=1000\digitize_bin_10\PCA_n=10.csv"
    data = read_data_from_PCA_digitized_file(data_address)
    
    _ , cols = np.shape(data)
    data_column_names = ['c' + str(i) for i in range(cols-1)]
    data_column_names.append('Person')
    
    #print(data_column_names)
    #print(data[0:10,:])
    target_column_name = 'Person'
    
    the_best_validation_strategy(data = data, data_column_names = data_column_names, target_column_name = target_column_name)
    
if __name__ == '__main__':
    
    #select_hyperparameters()
    test_the_best_validation_strategy()
    #cProfile.run('re.compile("kfoldcrossvalidationForBNModel_UsingPanda|10, data, target_column_name = "Person", scoring = "f1_micro"")')
    