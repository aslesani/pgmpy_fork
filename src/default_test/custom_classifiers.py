'''
Created on Jan 30, 2018

@author: Adele
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics.classification import f1_score

from Abdollahi import read_Abdoolahi_data

from DimensionReductionandBNStructureLearning import shift_each_column_separately, digitize_dataset
from read_write import read_data_from_CSV_file


def test_different_classifiers(data_features, data_target, k , shuffle, selected_classifiers):
    '''
    Parameters:
    

    '''
    #selected_classifiers = [12]
    names = ["1-Nearest Neighbors", 
             "2-Nearest Neighbors",
             "3-Nearest Neighbors",
             "4-Nearest Neighbors",
             "5-Nearest Neighbors",
             "Linear SVM", 
             "RBF SVM", 
             "Poly SVM degree = 3",
             "Poly SVM degree = 4",
             "Poly SVM degree = 5",
             "LinearSVC",
             "Gaussian Process",
             "Decision Tree",
             "Random Forest", 
             "Neural Net", 
             "AdaBoost",
             "Naive Bayes", 
             "QDA"
             ]
    
    classifiers = [
        KNeighborsClassifier(1),
        KNeighborsClassifier(2),
        KNeighborsClassifier(3),
        KNeighborsClassifier(4),
        KNeighborsClassifier(5),
        SVC(kernel="linear", C=1.0),
        SVC(kernel='rbf', gamma=0.7, C=1.0),
        SVC(kernel='poly', degree=3, C=1.0, gamma = 'auto'),
        SVC(kernel='poly', degree=4, C=1.0, gamma = 'auto'),
        SVC(kernel='poly', degree=5, C=1.0, gamma = 'auto'),
        LinearSVC(C=1.0),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]
    
    names = [names[i] for i in selected_classifiers]
    classifiers = [classifiers[i] for i in selected_classifiers]
    
    #split data to k gruop
    kf = KFold(n_splits=k, random_state=None, shuffle=shuffle)
    k_iter = -1
    scores = np.ndarray(shape = (len(classifiers) , k) , dtype = float)
    for train_index, test_index in kf.split(data_features):
       k_iter +=1
       #print("k_iter:" , k_iter)
       #print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = data_features[train_index], data_features[test_index]
       #print("X_train:" , X_train)
       y_train, y_test = data_target[train_index], data_target[test_index]
    # iterate over classifiers
       for name, clf, clasifier_index in zip(names, classifiers, range(len(classifiers))):
           #clf.fit(train_features , train_target.values.ravel())
           #print(name)
           clf.fit(X_train, y_train)
    
           predicted = clf.predict(X_test)#(test.loc[:, test.columns != 'Person'])
           #f_measure_score= f1_score(y_true = test[['Person']], y_pred = predicted, average = 'micro')#cross_val_score(clf, data, target, cv=10 , scoring='f1_macro') ####10-fold cross validation 
           scores[clasifier_index][k_iter]= f1_score(y_true = y_test, y_pred = predicted, average = 'micro')#cross_val_score(clf, data, target, cv=10 , scoring='f1_macro') ####10-fold cross validation 
    
           #print(name , " f_measure: " , f_measure_score)
            
    #print(scores)
    avg_f_score = np.mean(scores, axis = 1)
    #for n, avg in zip(names , avg_f_score):
     #   print(n , avg)
        
    return names , avg_f_score
    
def a_little_test():
    clf = KNeighborsClassifier(1)
    train = [
        [1,1,0] , 
        [0,0,1],
        [1,1,1]
        ]
    target = [1,0,1]
    model  = clf.fit(train , target)
    print(model.predict([0,0,1]))


def select_hyper_parameters_for_custom_classifiers_on_different_datasets(shuffle = True):
    
    data_address = r"E:\pgmpy\PCA on Bag of sensor events_no overlap\delta={delta}\PCA_n={n}.csv"

    deltas = [15,30,45,60,75,90,100, 120,150, 180,200,240,300,400,500,600,700,800,900,1000]
    n = range(2,11)
    
    best_hyper_parameters_for_classifiers = np.zeros((17, 2) , dtype = float)# 15 is number of classifiers
    #the first column is for best selected_n
    #the second column is for best selected_delta
    best_avg_f_scores = np.zeros(17 , dtype = float)
    
    #selected_delta = 75
    #selected_n = 9
    
    for cls_index in range(15,18):
        for selected_n in n:
            for selected_delta in deltas:
                #print(selected_n, selected_delta)
                data = digitize_dataset(data_address = data_address.format(delta = selected_delta, n = selected_n), selected_bin = 10, address_to_save = "", isSave=False , has_header = False , return_as_pandas_data_frame = False)
                data = shift_each_column_separately(data)
                
                names , avg_f_score = test_different_classifiers(data_features = data[:,0:selected_n], data_target = data[:,-1], k = 10,shuffle = shuffle, selected_classifiers = [cls_index])
                #print("##############",len(avg_f_score))
                if avg_f_score > best_avg_f_scores[cls_index]:
                    best_avg_f_scores[cls_index] = avg_f_score
                    best_hyper_parameters_for_classifiers[cls_index][0] = selected_n
                    best_hyper_parameters_for_classifiers[cls_index][1] = selected_delta
        
        print("classifier:", cls_index , "best_f_score:" , best_avg_f_scores[cls_index] , "best_hyper_parameters:" , best_hyper_parameters_for_classifiers[cls_index])

                    
    for i in range(len(best_avg_f_scores)):
        print(names[i] , "best_f_score:" , best_avg_f_scores[i] , "best_hyper_parameters:" , best_hyper_parameters_for_classifiers[i])
                

def test_test_different_classifiers(address_to_read):
    #for applying the iccke paper on CASAS dataset
    data = read_data_from_CSV_file(dest_file = address_to_read, data_type = int, has_header = False, return_as_pandas_data_frame = False , remove_date_and_time = False , return_header_separately = False , convert_int_columns_to_int = True)
    data_features = data[:,0:-2]
    #print(type(data_features))
    data_target = data[:,-2]
   
    names , avg_f_score = test_different_classifiers(data_features = data_features , data_target = data_target, k=10 , shuffle = True, selected_classifiers = [14])#[0,1,2,3,4,12,14])
     
    for i in range(len(avg_f_score)):
        print(names[i] , "best_f_score:" , avg_f_score[i])
   
def test_the_best_algorithm_of_iccke_paper(add_str_to_path):
    
    activity_bag_iccke = r"E:\pgmpy\{path}\Bag of sensor events_based on activities\based_on_activities_iccke_approach.csv"
    print(add_str_to_path)
    test_test_different_classifiers(address_to_read = activity_bag_iccke.format(path = add_str_to_path))
    

if __name__ == "__main__":
    
    test_the_best_algorithm_of_iccke_paper('Tulum2010')
    #select_hyper_parameters_for_custom_classifiers_on_different_datasets(shuffle = False)
    #a_little_test()
    #file_address = "E:\pgmpy\Bag of sensor events_no overlap_based on different deltas\delta_900min.csv"
    
    #data = read_Abdoolahi_data()
    #train = data[0:700] 
    #test = data[700:]
    #print(train.loc[:, train.columns != 'res'])
    #test_different_classifiers(train_features = train.loc[:, train.columns != 'Person'], train_target = train[['Person']], test = test)