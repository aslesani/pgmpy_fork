'''
Created on Aug 11, 2020

@author: Adele
'''

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score



def calculate_different_metrics(y_true , y_predicted):
    f1_score_micro = f1_score(y_true, y_predicted, average='micro') 
    f1_score_macro = f1_score(y_true, y_predicted, average='macro') 
    precision = precision_score(y_true, y_predicted, average='micro') 
    recall = recall_score(y_true, y_predicted, average='micro')
    accuracy = accuracy_score(y_true, y_predicted)
    
    scores = {'f1_score_micro': round(f1_score_micro,2) ,
              'f1_score_macro': round(f1_score_macro, 2) ,
              'precision' : round(precision,2) ,
              'recall' : round(recall,2) ,
              'accuracy' : round(accuracy ,2) 
              }
    
    return scores
    
if __name__ == '__main__':
    print(calculate_different_metrics([1, 2, 1, 2, 2, 2 ,2, 2, 2, 1], [1 ,2 ,1, 2, 2 ,2, 2 ,2, 2 ,1]))