'''
Created on Oct 27, 2017

@author: Adele
'''

import numpy as np
import pandas

data = pandas.read_csv("kaggle.csv")

data2 = data[["Survived", "Sex", "Pclass"]]
#data2 = data[["Survived", "Sex", "Pclass"]].replace(["female", "male"], [0, 1]).replace({"Pclass": {3: 0}})

intrain = np.random.rand(len(data2)) < 0.8

dtrain = data2[intrain]
dtest = data2[~intrain]

##print(len(dtrain), len(dtest))

from pgmpy.models import BayesianModel
titanic = BayesianModel()
titanic.add_edges_from([("Sex", "Survived"), ("Pclass", "Survived")])
titanic.fit(dtrain)
for cpd in titanic.get_cpds():
    print(cpd)


print(dtest[["Sex", "Pclass"]])
titanic.predict(dtest[["Sex", "Pclass"]])
