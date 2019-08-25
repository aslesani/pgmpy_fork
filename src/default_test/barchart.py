'''
Created on July 5, 2019

@author: Adele
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# data to plot
n_groups = 5
means_frank = (90, 55, 40, 65)
means_guido = (85, 62, 54, 20)

KNN = (0.808125, 0.91095571,0.87350746,0.87802198,0.95)
SVM = (0.845625, 0.91095571, 0.88992537, 0.87802198, 0.95)
DT = (0.845625, 0.91095571, 0.88992537, 0.87802198, 0.95)
RF = (0.845625, 0.91095571, 0.88992537, 0.87802198, 0.95)
NN = (0.845625, 0.91095571, 0.87502195, 0.87802198, 0.93)
Adaboost = (0.845625, 0.91095571, 0.88992537, 0.87802198, 0.93)
NB = (0.8425, 0.91095571, 0.88992537, 0.86263736, 0.91)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.8

rects = [0]* 7
i = 0
for classifier, color, label in zip([KNN, SVM, DT, RF, NN, Adaboost, NB], 
                                    ['navy','blue','r', 'yellow', 'orange', 'lightgreen', 'g'], 
									['KNN', 'SVM', 'DT', 'RF', 'NN', 'Adaboost', 'NB']):
									#['K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'Neural Network', 'Adaboost', 'Naive Bayes']):
    rects[i] = plt.bar(index + i * bar_width, classifier, bar_width,
    alpha=opacity,
    color=color,
    label=label)
    i+=1
'''
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Guido')
'''
font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=12)
									   
plt.xlabel('∆t', fontsize = 12, fontname = 'Times New Roman')
plt.ylabel('F-measure', fontsize = 12, fontname = 'Times New Roman')
plt.title('Classifiers', fontsize = 12, fontname = 'Times New Roman')
plt.xticks(index + 0.2, ('30', '75', '90', '700','900'))


plt.legend(prop=font, bbox_to_anchor=(1, 1))#loc = 'lower left')

plt.tight_layout()
plt.show()