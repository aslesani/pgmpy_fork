'''
Created on Jan 27, 2018

@author: Adele
'''

import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from Validation import calculate_different_metrics

values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
                       columns=['A', 'B', 'C', 'D', 'E'])
train_data = values[:800]
predict_data = values[800:]
model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
model.fit(values)
predict_data = train_data.copy()#predict_data.copy()
y_true = predict_data['E'].values
print(type(y_true))
print(y_true)
predict_data.drop('E', axis=1, inplace=True)
y_pred = model.predict(predict_data)
print(y_pred)
print(calculate_different_metrics(y_true = y_true, y_predicted = y_pred))

#print(calculate_different_metrics([1,1,1], y_predicted = [1,1,1]))