'''
Created on Feb 17, 2018

@author: cloud
'''
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_bar_chart(x_values, y_values , y_label , title):
    y_pos = np.arange(len(x_values))
     
    plt.bar(y_pos, y_values, align='center', alpha=0.5)
    plt.xticks(y_pos, x_values)
    plt.ylabel(y_label)
    plt.title(title)
 
    plt.show()

def read_column_and_person_from_Bag_of_sensor_events_based_on_activities():
    print("hi")

    f = open( r"E:\pgmpy\Bag of sensor events_based on activities\based_on_activities.csv","r")
    all_features = np.zeros((3216, 3), dtype= object )#np.str)1003 +1
    
    counter = 0
    for line in f:
        cells = line.split(',')
        all_features[counter][-1] = int(cells[-1].split('\\')[0])# activity
        all_features[counter][-2] = int(cells[-2]) # person
        all_features[counter][-3] = int(cells[-3]) # alaki

        counter +=1
        
    print(all_features[0])
    data = pd.DataFrame(all_features , columns = ['alaki' , 'person' , 'activity'])
    
    print(data.groupby(['person' , 'activity']).agg(['count']))

    
    
    
if __name__ == "__main__":
    read_column_and_person_from_Bag_of_sensor_events_based_on_activities()
     
