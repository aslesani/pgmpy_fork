'''
Created on Jan 30, 2018

@author: Adele
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
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

def test_different_classifiers(train_features ,train_target , test):
    '''
    Parameters:
    

    '''
    names = [#"1-Nearest Neighbors", 
             #"2-Nearest Neighbors",
             #"3-Nearest Neighbors",
             #"4-Nearest Neighbors",
             #"5-Nearest Neighbors",
             #"Linear SVM", 
             #"RBF SVM", 
             #"Poly SVM degree = 3",
             #"Poly SVM degree = 4",
             #"Poly SVM degree = 5",
             #"LinearSVC",
             #"Gaussian Process",
             "Decision Tree",
             "Random Forest", 
             "Neural Net", 
             "AdaBoost",
             "Naive Bayes", 
             "QDA"
             ]
    
    classifiers = [
        #KNeighborsClassifier(1),
        #KNeighborsClassifier(2),
        #KNeighborsClassifier(3),
        #KNeighborsClassifier(4),
        #KNeighborsClassifier(5),
        #SVC(kernel="linear", C=1.0),
        #SVC(kernel='rbf', gamma=0.7, C=1.0),
        #SVC(kernel='poly', degree=3, C=1.0),
        #SVC(kernel='poly', degree=4, C=1.0),
        #SVC(kernel='poly', degree=5, C=1.0),
        #LinearSVC(C=1.0),
        #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]
      
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(train_features , train_target.values.ravel())
        predicted = clf.predict(test.loc[:, test.columns != 'Person'])
        f_measure_score= f1_score(y_true = test[['Person']], y_pred = predicted, average = 'micro')#cross_val_score(clf, data, target, cv=10 , scoring='f1_macro') ####10-fold cross validation 
        print(name , " f_measure: " , f_measure_score)
        
        
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

if __name__ == "__main__":
    #a_little_test()
    data = read_Abdoolahi_data()
    train = data[0:700] 
    test = data[700:]
    #print(train.loc[:, train.columns != 'res'])
    test_different_classifiers(train_features = train.loc[:, train.columns != 'Person'], train_target = train[['Person']], test = test)