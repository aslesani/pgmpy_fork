'''
Created on Sep 26, 2018

@author: Adele
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from plot_results import plot_results

x1 = [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]
x2 = list(range(1,15)) + x1

Twor2009_BOE_TBWF = [80,80,92,90,87,91,90,88,88,86,89,88,83,89,91,92,89,93,96,86]
Twor2009_BOE_ABWF = [73] * len(x1)
Twor2009_BOE_HWF = [70,75,77,76,76,77,76,76,75,74,75,76,75,73,73,74,74,73,74,73]

Twor2009_SOE_TBWF = [87,80,81,76,75,73,75,72,67,65,66,63,58,45,55,53,52,46,50,47]
Twor2009_SOE_ABWF = [87] * len(x1)
Twor2009_SOE_HWF = [89,88,88,87,87,88,87,87,87,86,87,86,87,87,86,86,87,86,86,86]

#####################################

Tulum2009_BOE_TBWF = [96,94,93,92,90,90,90,88,87,86,86,83,82,80,78,76,72,69,64,63]
Tulum2009_BOE_ABWF = [100] * len(x1)
Tulum2009_BOE_HWF = [99,99,99,98,98,93,92,92,90,88,88,88,86,86,80,80,80,80,80,80]

Tulum2009_SOE_TBWF = [92,93,92,90,91,88,89,87,86,85,85,85,83,81,77,76,59,74,70,70]
Tulum2009_SOE_ABWF = [92] * len(x1)
Tulum2009_SOE_HWF = [89,88,94,89,94,89,93,93,93,89,94,90,90,94,90,94,93,94,89,93]

###################################

Tulum2010_BOE_TBWF = [71,70,70,70,69,69,69,68,68,67,67,67,67,65,65,65,68,64,61,60]
Tulum2010_BOE_ABWF = [99] * len(x1)
Tulum2010_BOE_HWF =  [99,99,99,98,98,93,92,92,90,88,88,88,86,86,80,80,80,80,80,80]

Tulum2010_SOE_TBWF = [76,78,80,80,80, 82,82,83,84,84,84,86,86,84,84,81,80,79,77,75]
Tulum2010_SOE_ABWF = [90] * len(x1)
Tulum2010_SOE_HWF = [84,84,84,85,88,85,85,84,88,85,85,88,88,88,88,88,86,90,92,90]

#***********************************
#***********************************
layer_numbers = [1,2,3,4]

Twor2009_TBWF_LSTM_1layer_val_acc = [94.28,94.93,93.44,94.72,94.39,93.91,92.83,95.23,90.52] + [92.72, 94.52, 94.01, 92.67, 91.4, 92.31, 92.83, 91.38, 92.97, 82.05, 87.41, 80.65, 86.11, 76.09, 86.08, 76.71, 84.13, 72.22, 72.09, 77.78, 67.74, 71.43, 75.0, 71.43, 70.0]
Twor2009_TBWF_LSTM_2layer_val_acc = [92.52,92.24,94.44,94.79,95.48,93.50,94.73,95.96,95.72] + [92.3, 94.37, 94.66, 95.57, 93.37, 90.62, 91.9, 90.95, 93.51, 89.74, 91.11, 75.81, 76.85, 79.35, 81.01, 75.34, 82.54, 74.07, 74.42, 75.0, 77.42, 71.43, 83.33, 61.9, 70.0]
Twor2009_TBWF_LSTM_3layer_val_acc = [90.01,93.27,94.37, 95.35,93.04,95.03,95.07,96.21,94.65] + [95.15, 93.61, 95.47, 93.36, 93.01, 92.68, 89.1, 91.81, 88.11, 83.33, 85.19, 83.06, 82.41, 84.78, 86.08, 87.67, 85.71, 68.52, 74.42, 83.33, 61.29, 85.71, 83.33, 71.43, 70.0]
Twor2009_TBWF_LSTM_4layer_val_acc = [96.52, 95.34,90.60,93.93,93.22,95.63,92.94,95.72,94.52] + [93.01, 93.46, 96.12, 95.57, 94.98, 94.37, 90.97, 86.64, 84.32, 82.69, 82.22, 77.42, 75.93, 72.83, 82.28, 71.23, 80.95, 83.33, 76.74, 75.0, 70.97, 71.43, 58.33, 71.43, 70.0]
####################################

Twor2009_ABWF_LSTM_1layer_val_acc = [71.58] * len(x2)
Twor2009_ABWF_LSTM_2layer_val_acc = [74.84] * len(x2)
Twor2009_ABWF_LSTM_3layer_val_acc = [79.97] * len(x2)
Twor2009_ABWF_LSTM_4layer_val_acc = [74.84] * len(x2)

####################################
Twor2009_HWF_LSTM_1layer_val_acc = [89.44, 92.61, 90.69, 90.92, 94.67, 90.41, 85.95, 89.57, 88.28] + [90.15, 88.72, 88.39, 87.13, 87.91, 87.65, 88.81, 86.17, 83.9, 78.2, 80.41, 80.06, 80.68, 76.09, 77.68, 77.55, 64.78, 73.56, 72.34, 72.89, 70.63, 64.71, 70.23, 73.8, 72.25]
Twor2009_HWF_LSTM_2layer_val_acc = [91.15, 91.26, 91.81, 91.03, 91.83, 90.54, 90.87, 89.13, 88.51] + [91.1, 90.7, 88.47, 88.26, 86.93, 89.19, 86.69, 87.41, 80.91, 85.87, 81.66, 80.89, 82.53, 81.01, 80.91, 68.24, 78.96, 78.1, 76.9, 73.2, 77.74, 71.98, 60.78, 74.73, 77.52]
Twor2009_HWF_LSTM_3layer_val_acc = [92.65, 91.26, 92.06, 92.8, 94.07, 91.13, 92.46, 89.78,91.64] + [90.79, 89.55, 91.02, 89.04, 88.98, 90.1, 83.11, 87.65, 85.58, 84.39, 84.97, 83.12, 83.66, 82.03, 80.32, 81.83, 80.45, 79.15, 77.66, 64.01, 78.52, 77.09, 71.16, 71.63, 79.38]
Twor2009_HWF_LSTM_4layer_val_acc = [94.69, 93.3, 91.81, 93.85, 91.34, 93.15, 89.55, 91.39, 88.05] + [91.5, 90.62, 89.66, 88.78, 89.33, 89.55, 88.59, 88.89, 85.71, 85.06, 85.93, 87.31, 78.12, 76.81, 74.89, 77.7, 72.84, 73.26, 77.2, 77.95, 72.18, 76.01, 78.45, 81.24, 75.5]

#**********************************
#**********************************

Twor2009_TBWF_LSTM_1layer_val_f1 = [93.94, 97.25, 97.42, 97.13, 96.37, 97.15, 96.4, 98.16, 96.79, 96.53, 97.36, 97.2, 97.32, 94.9, 97.36, 95.88, 93.25, 93.35, 93.01, 94.5, 95.68, 93.05, 91.06, 90.96, 83.98, 92.24, 87.44, 91.08, 81.75, 78.43, 85.71, 97.87, 83.33, 78.79]
Twor2009_TBWF_LSTM_2layer_val_f1 = [94.84, 95.62, 95.82, 96.92, 97.23, 96.66, 97.62, 97.35, 96.37, 97.01, 97.17, 97.99, 96.83, 95.4, 95.71, 95.89, 98.66, 96.02, 96.33, 92.37, 89.5, 94.1, 92.36, 89.48, 79.17, 93.19, 79.99, 83.66, 81.75, 85.19, 83.33, 88.37, 80.0, 75.0]
Twor2009_TBWF_LSTM_3layer_val_f1 = [94.76, 94.91, 96.21, 95.86, 95.79, 98.72, 97.34, 98.48, 96.52, 98.08, 97.41, 96.75, 97.5, 96.08, 96.52, 90.12, 97.54, 99.73, 99.35, 94.9, 91.44, 90.86, 90.36, 81.91, 83.1, 98.36, 79.99, 95.04, 85.71, 85.19, 78.26, 85.71, 76.47, 82.35]
Twor2009_TBWF_LSTM_4layer_val_f1 = [93.87, 96.64, 97.52, 96.88, 96.72, 97.26, 98.36, 97.35, 98.1, 97.15, 95.87, 97.12, 95.71, 95.89, 97.11, 96.72, 93.49, 87.82, 95.95, 92.81, 86.63, 96.63, 89.79, 89.4, 85.02, 91.26, 88.65, 85.0, 98.59, 68.09, 90.2, 95.65, 76.47, 78.79]
####################################

Twor2009_ABWF_LSTM_1layer_val_f1 = [84.13] * len(x2)
Twor2009_ABWF_LSTM_2layer_val_f1 = [85.21] * len(x2)
Twor2009_ABWF_LSTM_3layer_val_f1 = [85.51] * len(x2)
Twor2009_ABWF_LSTM_4layer_val_f1 = [89.40] * len(x2)

####################################
Twor2009_HWF_LSTM_1layer_val_f1 = [94.27, 94.26, 94.97, 94.72, 96.32, 95.93, 93.47, 93.89, 93.81, 93.96, 90.9, 92.88, 91.91, 91.74, 92.34, 92.06, 90.73, 90.33, 88.87, 84.43, 88.06, 87.11, 86.47, 85.57, 86.46, 86.62, 86.98, 83.7, 75.58, 82.89, 87.18, 77.01, 83.94, 76.62]
Twor2009_HWF_LSTM_2layer_val_f1 = [94.51, 94.48, 95.08, 93.22, 96.97, 95.73, 93.72, 94.44, 92.59, 94.13, 92.9, 93.62, 92.86, 92.81, 91.51, 91.35, 90.42, 89.93,89.46, 88.99, 88.68, 86.23, 87.02, 84.06, 86.41, 87.87, 85.53, 82.54, 81.57, 80.59, 86.49, 81.24, 85.72, 84.71]
Twor2009_HWF_LSTM_3layer_val_f1 = [95.10, 94.94, 95.22, 94.92, 95.5, 94.83, 94.82, 94.66, 94.39, 93.91, 93.4, 94.41, 93.61, 92.6, 92.74, 93.52, 89.46, 90.28, 92.26, 87.13, 87.95, 86.01, 87.47, 88.04, 87.74, 85.15, 87.61, 88.65, 88.87, 88.05, 87.36, 86.74, 83.65, 80.8]
Twor2009_HWF_LSTM_4layer_val_f1 = [94.47, 94.28, 93.68, 95.06, 95.89, 94.54, 93.07, 91.52, 94.11, 93.72, 92.83, 94.48, 93.19, 93.85, 91.58, 86.03, 87.68, 83.22, 89.4, 78.52, 75.18, 78.49, 80.08, 62.45, 87.35, 75.0, 60.64, 77.23, 54.4, 66.43, 81.37, 60.73, 65.22, 61.57]


#**********************************
#**********************************

Twor2009_TBWF_RNN_1layer_val_f1 = [96.72, 96.18, 95.24, 96.45, 95.76, 96.76, 95.55, 96.82, 97.17, 96.53, 97.82, 97.12, 95.98, 96.18, 96.64, 95.0, 93.02, 94.27, 96.68, 93.15, 95.75, 93.57, 97.77, 95.34, 87.63, 95.8, 89.8, 85.26, 79.96, 70.83, 78.26, 70.27, 83.33, 75.0]
Twor2009_TBWF_RNN_2layer_val_f1 = [95.00, 96.7, 97.42, 97.45, 95.94, 96.59, 96.07, 97.09, 97.96, 96.92, 97.08, 97.74, 95.89, 97.46, 96.33, 96.37, 95.89, 95.15, 98.36, 95.65, 98.35, 89.72, 91.73, 91.72, 94.86, 85.45, 86.31, 83.66, 81.75, 85.19, 92.31, 16.0, 76.47, 85.71]
Twor2009_TBWF_RNN_3layer_val_f1 = [94.68, 95.46, 96.28, 98.26, 96.99, 96.23, 98.31, 96.24, 97.89, 97.4, 96.4, 95.55, 97.41, 95.89, 96.52, 95.89, 97.09, 96.91, 95.48, 93.17, 92.11, 97.1, 93.01, 97.4, 95.7, 97.5, 86.31, 89.64, 64.14, 91.23, 90.2, 73.68, 89.47, 75.0] 
Twor2009_TBWF_RNN_4layer_val_f1 = [92.84, 94.5, 96.85, 97.18, 96.5, 97.05, 97.0, 97.96, 97.74, 97.4, 96.84, 98.16, 96.68, 95.4, 96.94, 96.06, 96.62, 94.09, 91.5, 94.9, 97.94, 96.63, 97.2, 97.36, 84.96, 82.17, 95.15, 82.0, 77.85, 78.43, 44.44, 40.0, 76.47, 70.97]
####################################

Twor2009_ABWF_RNN_1layer_val_f1 = [84.23] * len(x2)
Twor2009_ABWF_RNN_2layer_val_f1 = [81.79] * len(x2)
Twor2009_ABWF_RNN_3layer_val_f1 = [81.75] * len(x2)
Twor2009_ABWF_RNN_4layer_val_f1 = [82.31] * len(x2)

####################################
Twor2009_HWF_RNN_1layer_val_f1 = [95.47, 93.0, 93.99, 93.89, 92.75, 93.55, 92.12, 91.51, 93.25, 91.08, 91.52, 92.04, 91.38, 89.49, 90.92, 89.4, 87.05, 87.75, 85.82, 85.04, 83.31, 83.98, 83.58, 82.45, 82.95, 82.13, 81.32, 82.59, 83.28, 79.27, 81.02, 80.27, 79.38, 81.19]
Twor2009_HWF_RNN_2layer_val_f1 = [92.76, 94.78, 93.72, 93.91, 92.79, 94.64, 91.95, 91.87, 92.12, 92.0, 91.29, 91.36, 91.1, 91.14, 91.14, 90.4, 88.85, 86.05, 86.49, 83.09, 85.18, 84.6, 83.17, 83.39, 84.45, 82.48, 81.5, 81.99, 78.91, 82.81, 80.89, 77.91, 80.81, 80.22]
Twor2009_HWF_RNN_3layer_val_f1 = [89.49, 94.53, 93.63, 93.85, 93.02, 92.99, 93.12, 92.81, 93.67, 91.58, 92.43, 92.05, 90.32, 92.54, 88.05, 87.37, 82.7, 89.62, 89.56, 83.37, 84.38, 83.62, 82.45, 84.25, 83.55, 85.01, 83.75, 77.54, 83.14, 81.53, 82.8, 80.73, 82.43, 80.72]
Twor2009_HWF_RNN_4layer_val_f1 = [93.68, 93.35, 94.39, 95.53, 92.97, 94.37, 92.26, 92.48, 92.56, 91.76, 91.71, 93.39, 89.8, 91.95, 89.86, 90.36, 87.24, 89.44, 83.45, 85.29, 85.08, 87.09, 81.56, 82.1, 85.09, 76.4, 83.51, 85.84, 79.85, 77.65, 78.88, 50.14, 67.6, 83.71]

#*********************************
#*********************************

#**********************************
#**********************************

Tulum2009_TBWF_LSTM_1layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_LSTM_2layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_LSTM_3layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_LSTM_4layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
####################################

Tulum2009_ABWF_LSTM_1layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_LSTM_2layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_LSTM_3layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_LSTM_4layer_val_acc = [99.96] * len(x2)

####################################
Tulum2009_HWF_LSTM_1layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_LSTM_2layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_LSTM_3layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_LSTM_4layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]

#**********************************
#**********************************

Tulum2009_TBWF_RNN_1layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_RNN_2layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_RNN_3layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
Tulum2009_TBWF_RNN_4layer_val_acc = [80.46, 80.79, 81.04, 81.5, 80.7, 80.7, 79.42, 80.37, 80.56, 80.06, 80.95, 80.44, 79.93, 80.0, 80.22, 71.43, 64.39, 58.41, 53.47, 48.91, 45.35, 40.51, 34.72, 29.85, 25.4, 18.97, 6.0, 0, 0, 0, 0, 0, 0, 0]
####################################

Tulum2009_ABWF_RNN_1layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_RNN_2layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_RNN_3layer_val_acc = [99.96] * len(x2)
Tulum2009_ABWF_RNN_4layer_val_acc = [99.96] * len(x2)

####################################
Tulum2009_HWF_RNN_1layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_RNN_2layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_RNN_3layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]
Tulum2009_HWF_RNN_4layer_val_acc = [90.5, 93.41, 94.89, 95.84, 96.27, 96.7, 96.83, 97.24, 97.47, 97.61, 97.86, 97.92, 97.97, 98.09, 98.19, 98.31, 98.33, 98.33, 98.33, 98.32, 98.32, 98.32, 98.32, 98.32, 98.32, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31, 98.31]

#************************************
#************************************

Tulum2010_TBWF_LSTM_1layer_val_acc = []
Tulum2010_TBWF_LSTM_2layer_val_acc = []
Tulum2010_TBWF_LSTM_3layer_val_acc = []
Tulum2010_TBWF_LSTM_4layer_val_acc = []
####################################

Tulum2010_ABWF_LSTM_1layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_LSTM_2layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_LSTM_3layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_LSTM_4layer_val_acc = [100.00] * len(x2)

####################################
Tulum2010_HWF_LSTM_1layer_val_acc = []
Tulum2010_HWF_LSTM_2layer_val_acc = []
Tulum2010_HWF_LSTM_3layer_val_acc = []
Tulum2010_HWF_LSTM_4layer_val_acc = []

#**********************************
#**********************************

Tulum2010_TBWF_RNN_1layer_val_acc = []
Tulum2010_TBWF_RNN_2layer_val_acc = []
Tulum2010_TBWF_RNN_3layer_val_acc = []
Tulum2010_TBWF_RNN_4layer_val_acc = []
####################################

Tulum2010_ABWF_RNN_1layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_RNN_2layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_RNN_3layer_val_acc = [100.00] * len(x2)
Tulum2010_ABWF_RNN_4layer_val_acc = [100.00] * len(x2)

####################################
Tulum2010_HWF_RNN_1layer_val_acc = []
Tulum2010_HWF_RNN_2layer_val_acc = []
Tulum2010_HWF_RNN_3layer_val_acc = []
Tulum2010_HWF_RNN_4layer_val_acc = []

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#**********************************
#**********************************

Tulum2009_TBWF_LSTM_1layer_val_acc_with_shuffle = [96.01, 96.17, 96.79, 97.11, 96.52, 96.32, 95.63, 97.23, 96.46, 95.01, 96.43, 96.21, 97.32, 97.5, 95.9, 96.43, 90.15, 89.38, 91.09, 90.22, 87.21, 87.34, 83.33, 80.6, 88.89, 86.21, 86.0, 84.44, 80.95, 82.5, 84.85, 67.74, 70.37, 64.0]
Tulum2009_TBWF_LSTM_2layer_val_acc_with_shuffle = [96.22, 96.76, 96.68, 96.72, 96.2, 97.98, 94.8, 95.15, 95.96, 93.35, 96.43, 95.9, 95.99, 96.79, 92.91, 94.64, 92.42, 94.69, 95.05, 91.3, 89.53, 91.14, 88.89, 91.04, 85.71, 81.03, 66.0, 73.33, 71.43, 75.0, 75.76, 77.42, 59.26, 68.0]
Tulum2009_TBWF_LSTM_3layer_val_acc_with_shuffle = [96.42, 95.88, 96.58, 96.85, 94.46, 94.85, 97.3, 95.15, 95.96, 96.68, 96.43, 95.58, 97.99, 97.14, 97.01, 91.67, 91.67, 95.58, 94.06, 85.87, 86.05, 92.41, 84.72, 79.1, 80.95, 84.48, 78.0, 82.22, 66.67, 77.5, 63.64, 74.19, 88.89, 72.0]
Tulum2009_TBWF_LSTM_4layer_val_acc_with_shuffle = [96.46, 96.1, 96.17, 97.11, 95.25, 93.75, 95.43, 96.54, 97.98, 95.57, 97.02, 95.58, 94.65, 97.14, 97.01, 94.05, 90.91, 92.92, 88.12, 90.22, 94.19, 87.34, 84.72, 82.09, 84.13, 89.66, 76.0, 75.56, 71.43, 70.0, 75.76, 70.97, 66.67, 52.0]
####################################

Tulum2009_ABWF_LSTM_1layer_val_acc_with_shuffle = [100.00] * len(x2)
Tulum2009_ABWF_LSTM_2layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2009_ABWF_LSTM_3layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2009_ABWF_LSTM_4layer_val_acc_with_shuffle = [100] * len(x2)

####################################
Tulum2009_HWF_LSTM_1layer_val_acc_with_shuffle = [98.0, 98.74, 99.08, 99.17, 98.93, 99.31, 99.2, 99.61, 99.67, 99.4, 99.47, 99.63, 99.7, 99.66, 99.76, 99.54, 99.65, 99.72, 99.75, 99.64, 99.82, 99.5, 99.68, 99.57, 99.64, 99.75, 99.68, 99.68, 99.64, 99.57, 99.64, 99.68, 99.75, 99.78]
Tulum2009_HWF_LSTM_2layer_val_acc_with_shuffle = [97.98, 98.59, 98.74, 99.32, 99.14, 99.47, 99.39, 99.48, 99.64, 99.73, 99.53, 99.66, 99.63, 99.66, 99.69, 99.65, 99.54, 99.64, 99.61, 99.61, 99.64, 99.5, 99.61, 99.71, 99.71, 99.64, 99.64, 99.82, 99.61, 99.57, 99.82, 99.61, 99.5, 99.75]
Tulum2009_HWF_LSTM_3layer_val_acc_with_shuffle = [98.22, 98.69, 98.8, 99.44, 99.3, 99.31, 99.3, 99.54, 99.57, 99.6, 99.63, 99.5, 99.7, 99.69, 99.59, 99.79, 99.68, 99.68, 99.47, 99.61, 99.61, 99.71, 99.68, 99.5, 99.64, 99.68, 99.5, 99.78, 99.71, 99.57, 99.78, 99.53, 99.68, 99.82]
Tulum2009_HWF_LSTM_4layer_val_acc_with_shuffle = [98.24, 98.54, 98.72, 99.17, 99.3, 99.37, 99.52, 99.28, 99.38, 99.67, 99.7, 99.9, 99.46, 99.59, 99.59, 99.65, 99.79, 99.68, 99.54, 99.68, 99.57, 99.64, 99.57, 99.64, 99.82, 99.43, 99.68, 99.53, 99.71, 99.78, 99.61, 99.53, 99.61, 99.68]

#**********************************
#**********************************

Tulum2009_TBWF_RNN_1layer_val_acc_with_shuffle = [95.93, 96.84, 95.65, 95.67, 97.31, 96.32, 95.43, 95.84, 95.45, 96.12, 94.64, 95.58, 95.32, 96.43, 98.13, 96.43, 93.18, 91.15, 88.12, 90.22, 89.53, 81.01, 84.72, 89.55, 88.89, 77.59, 90.0, 88.89, 76.19, 67.5, 63.64, 74.19, 70.37, 64.0]
Tulum2009_TBWF_RNN_2layer_val_acc_with_shuffle = [97.16, 97.13, 96.48, 97.11, 96.2, 96.69, 96.05, 96.3, 96.46, 96.12, 96.73, 98.42, 96.99, 96.43, 95.52, 93.45, 90.91, 92.04, 93.07, 90.22, 90.7, 86.08, 88.89, 86.57, 82.54, 81.03, 76.0, 80.0, 83.33, 75.0, 72.73, 80.65, 74.07, 76.0]
Tulum2009_TBWF_RNN_3layer_val_acc_with_shuffle = [96.13, 97.72, 95.96, 95.67, 97.15, 96.69, 96.05, 96.54, 94.44, 96.12, 94.35, 97.48, 93.98, 95.36, 96.64, 91.07, 93.94, 89.38, 91.09, 93.48, 89.53, 92.41, 91.67, 91.04, 82.54, 87.93, 76.0, 75.56, 76.19, 85.0, 72.73, 64.52, 74.07, 52.0]
Tulum2009_TBWF_RNN_4layer_val_acc_with_shuffle = [96.13, 96.32, 96.79, 96.19, 96.52, 96.69, 95.84, 94.69, 96.72, 96.12, 95.83, 96.53, 97.66, 94.29, 97.39, 97.62, 90.15, 89.38, 88.12, 86.96, 88.37, 92.41, 88.89, 88.06, 85.71, 87.93, 82.0, 75.56, 66.67, 72.5, 75.76, 61.29, 74.07, 48.0]
####################################

Tulum2009_ABWF_RNN_1layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2009_ABWF_RNN_2layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2009_ABWF_RNN_3layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2009_ABWF_RNN_4layer_val_acc_with_shuffle = [100] * len(x2)

####################################

Tulum2009_HWF_RNN_1layer_val_acc_with_shuffle = [98.26, 98.49, 98.94, 99.26, 99.33, 99.47, 99.33, 99.35, 99.41, 99.7, 99.63, 99.76, 99.53, 99.59, 99.8, 99.68, 99.75, 99.72, 99.75, 99.82, 99.68, 99.57, 99.82, 99.86, 99.75, 99.57, 99.82, 99.64, 99.53, 99.64, 99.61, 99.75, 99.82, 99.78]
Tulum2009_HWF_RNN_2layer_val_acc_with_shuffle = [98.28, 98.66, 98.91, 99.26, 98.87, 99.25, 99.42, 99.54, 99.44, 99.64, 99.7, 99.56, 99.7, 99.52, 99.76, 99.79, 99.82, 99.61, 99.72, 99.75, 99.57, 99.61, 99.54, 99.46, 99.61, 99.64, 99.64, 99.78, 99.64, 99.46, 99.5, 99.78, 99.53, 99.64]
Tulum2009_HWF_RNN_3layer_val_acc_with_shuffle = [97.92, 98.99, 99.05, 99.03, 99.36, 99.18, 99.26, 99.32, 99.57, 99.47, 99.73, 99.7, 99.56, 99.69, 99.62, 99.54, 99.75, 99.61, 99.72, 99.71, 99.5, 99.46, 99.64, 99.82, 99.61, 99.75, 99.71, 99.68, 99.78, 99.61, 99.75, 99.61, 99.71, 99.71]
Tulum2009_HWF_RNN_4layer_val_acc_with_shuffle = [98.02, 98.81, 99.19, 99.14, 99.2, 99.31, 99.26, 99.41, 99.44, 99.64, 99.63, 99.63, 99.66, 99.59, 99.66, 99.61, 99.61, 99.68, 99.68, 99.64, 99.64, 99.61, 99.75, 99.68, 99.64, 99.75, 99.57, 99.61, 99.71, 99.57, 99.5, 99.46, 99.53, 99.57]

#************************************
#************************************

Tulum2010_TBWF_LSTM_1layer_val_acc_with_shuffle = [84.62, 81.14, 78.68, 78.05, 75.64, 74.62, 74.19, 75.07, 73.71, 75.08, 73.65, 70.7, 72.12, 72.21, 72.3, 70.73, 67.45, 67.9, 67.61, 63.95, 70.0, 71.18, 67.17, 68.93, 63.42, 63.47, 62.01, 63.89, 68.64, 67.01, 70.24, 60.53, 77.14, 56.72]
Tulum2010_TBWF_LSTM_2layer_val_acc_with_shuffle = [85.33, 81.81, 79.41, 77.7, 75.62, 75.08, 74.92, 73.79, 70.57, 73.29, 73.93, 73.64, 72.51, 73.17, 70.4, 71.95, 67.88, 65.84, 65.12, 70.93, 68.94, 67.17, 65.35, 65.36, 63.04, 69.41, 65.92, 65.28, 73.73, 55.67, 70.24, 68.42, 50.0, 59.7]
Tulum2010_TBWF_LSTM_3layer_val_acc_with_shuffle = [84.69, 79.26, 76.91, 75.68, 76.15, 75.1, 75.13, 74.9, 74.23, 74.33, 72.88, 73.57, 72.63, 71.88, 70.54, 70.96, 68.63, 69.41, 67.11, 66.09, 69.36, 70.68, 70.82, 69.29, 69.26, 72.15, 68.72, 66.67, 63.56, 64.95, 71.43, 71.05, 67.14, 55.22]
Tulum2010_TBWF_LSTM_4layer_val_acc_with_shuffle = [85.72, 81.27, 76.09, 76.37, 74.45, 76.09, 75.03, 74.79, 75.11, 73.26, 73.23, 74.24, 72.2, 72.88, 71.42, 68.52, 67.77, 66.12, 66.11, 65.12, 67.45, 67.92, 64.74, 68.21, 66.15, 74.89, 68.72, 70.14, 61.02, 59.79, 69.05, 71.05, 62.86, 62.69]
####################################

Tulum2010_ABWF_LSTM_1layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_LSTM_2layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_LSTM_3layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_LSTM_4layer_val_acc_with_shuffle = [100] * len(x2)

####################################
Tulum2010_HWF_LSTM_1layer_val_acc_with_shuffle = [97.32,97.83,98.16,98.41,98.58,98.62,98.74,98.86,98.99,99.03,99.05,99.14,99.09,99.26,99.23] + [99.03, 98.93, 98.73, 99.13, 99.03, 98.53, 99.23, 98.93, 98.83, 98.63, 99.13, 98.63, 99.13, 99.23, 99.03, 98.83, 99.13, 99.03, 98.93]
Tulum2010_HWF_LSTM_2layer_val_acc_with_shuffle = [97.27,97.83,98.18,98.37,98.63,98.61,98.83,98.87,98.98,98.94,99.05,99.13,99.17,99.15,99.27] + [98.97, 99.17, 99.07, 98.97, 98.57, 98.57, 98.77, 98.77, 98.67, 98.97, 98.57, 99.07, 98.77, 98.67, 99.27, 99.27, 98.87, 98.87, 99.07]
Tulum2010_HWF_LSTM_3layer_val_acc_with_shuffle = [96.92,97.90,98.13,98.35,98.53,98.64,98.75,98.86,98.89,99.00,99.07,99.07,99.17,99.17,99.21] + [99.11, 98.91, 98.81, 99.11, 99.01, 98.51, 98.81, 99.01, 99.11, 99.21, 98.51, 98.81, 98.81, 98.71, 98.61, 98.91, 98.51, 99.21, 98.71]
Tulum2010_HWF_LSTM_4layer_val_acc_with_shuffle = [96.98,97.82,98.16,98.39,98.41,98.58,98.75,98.87,98.88,99.01,99.00,99.08,99.17,99.03,99.16] + [99.06, 98.46, 98.66, 98.56, 99.16, 98.96, 98.96, 98.66, 99.06, 99.16, 98.56, 99.16, 98.96, 99.06, 98.96, 98.86, 99.06, 98.56, 98.96]

#**********************************
#**********************************

Tulum2010_TBWF_RNN_1layer_val_acc_with_shuffle = [84.96, 82.08, 79.12, 80.1, 80.03, 78.51, 78.98, 77.91, 76.65, 76.81, 77.35, 77.62, 73.94, 73.88, 74.32, 70.58, 75.48, 73.8, 65.95, 67.05, 72.55, 74.44, 67.48, 67.5, 63.81, 67.12, 68.72, 65.28, 66.1, 62.89, 67.86, 63.16, 62.86, 67.16]
Tulum2010_TBWF_RNN_2layer_val_acc_with_shuffle = [85.45, 82.71, 81.37, 80.15, 79.87, 76.68, 78.58, 75.93, 77.19, 78.9, 77.21, 77.92, 78.06, 76.35, 74.49, 71.11, 75.8, 74.9, 70.27, 69.77, 71.7, 66.17, 70.21, 70.71, 69.26, 68.95, 63.13, 63.89, 63.56, 64.95, 66.67, 63.16, 60.0, 71.64]
Tulum2010_TBWF_RNN_3layer_val_acc_with_shuffle = [85.16, 83.11, 79.82, 81.46, 80.94, 75.97, 78.73, 77.72, 78.07, 78.31, 73.93, 72.04, 77.62, 76.56, 74.32, 75.15, 72.81, 75.45, 73.26, 73.45, 70.43, 69.67, 70.82, 66.43, 71.6, 64.38, 64.25, 63.19, 63.56, 59.79, 60.71, 67.11, 55.71, 59.7]
Tulum2010_TBWF_RNN_4layer_val_acc_with_shuffle = [84.32, 83.07, 81.16, 81.31, 80.39, 75.9, 75.61, 76.49, 78.55, 76.32, 71.97, 75.28, 74.1, 71.33, 75.9, 67.99, 69.27, 63.79, 69.6, 70.54, 68.3, 65.91, 66.57, 63.57, 70.43, 62.56, 70.95, 65.28, 63.56, 69.07, 61.9, 57.89, 42.86, 47.76]
####################################

Tulum2010_ABWF_RNN_1layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_RNN_2layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_RNN_3layer_val_acc_with_shuffle = [100] * len(x2)
Tulum2010_ABWF_RNN_4layer_val_acc_with_shuffle = [100] * len(x2)

####################################
Tulum2010_HWF_RNN_1layer_val_acc_with_shuffle = [97.10,97.73,98.17,98.40,98.58,98.64,98.74,98.88,98.76,99.05,99.05,99.17,99.10,99.27,99.24] + [99.24, 98.94, 98.54, 98.64, 99.04, 98.74, 99.14, 98.94, 98.54, 99.24, 98.54, 98.84, 98.84, 98.64, 99.04, 99.14, 99.14, 99.04, 99.24]
Tulum2010_HWF_RNN_2layer_val_acc_with_shuffle = [97.30,97.57,98.21,98.36,98.56,98.71,98.80,98.93,98.92,98.97,99.13,99.16,99.22,99.16,99.27] + [98.97, 99.17, 98.77, 98.57, 99.17, 98.87, 98.97, 98.67, 99.27, 98.57, 99.27, 98.57, 98.77, 99.07, 98.97, 99.17, 98.67, 98.67, 98.97]
Tulum2010_HWF_RNN_3layer_val_acc_with_shuffle = [97.02,97.80,98.13,98.34,98.52,98.67,98.55,98.93,98.91,99.01,99.08,99.16,99.23,99.05,99.18] + [98.98, 98.88, 98.78, 98.58, 99.08, 98.48, 98.58, 98.78, 99.08, 99.08, 99.18, 98.48, 98.68, 99.08, 98.88, 99.18, 98.48, 98.78, 98.48]
Tulum2010_HWF_RNN_4layer_val_acc_with_shuffle = [97.26,97.82,98.22,98.31,98.60,98.72,98.62,98.84,98.96,99.06,99.05,99.04,99.16,99.23,99.23] + [98.63, 98.53, 99.03, 98.53, 98.83, 98.53, 99.23, 98.53, 98.93, 99.13, 99.03, 99.13, 99.03, 99.03, 98.63, 98.73, 98.83, 98.73, 99.13]



def plot_six_plots(x1, BOE_TBWF,BOE_ABWF,BOE_HWF, SOE_TBWF,SOE_ABWF,SOE_HWF):
    
    x_label = '∆t', 
    y_label = 'F-measure' , 
    plt.xlabel('∆t' , fontsize = 16, fontname = 'Times New Roman')
    plt.ylabel('F-measure', fontsize = 16, fontname = 'Times New Roman')
        
    plt.xticks(fontsize = 14, fontname = "Times New Roman")
    plt.yticks(fontsize = 14, fontname = "Times New Roman")
    
    plt.plot(x1, BOE_TBWF, 'bx-', label = 'BOE+TBWF', linewidth = 2)
    plt.plot(x1, BOE_ABWF, 'go-', label = 'BOE+ABWF', linewidth = 2)
    plt.plot(x1,BOE_HWF, 'r+-' , label = 'BOE+HWF',linewidth = 2)
    
    plt.plot(x1, SOE_TBWF, 'b*--', label = 'SOE+TBWF', linewidth = 2)
    plt.plot(x1, SOE_ABWF, 'gx--', label = 'SOE+ABWF', linewidth = 2)
    plt.plot(x1,SOE_HWF, 'ro--' , label = 'SOE+HWF',linewidth = 2)
    
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=14)
    #plt.legend(prop=font, loc = 'lower left')#fontsize = 12, fontname = "Times New Roman")
    plt.show()
    '''
    y_val = [BOE_TBWF,BOE_ABWF,BOE_HWF]
    
    plot_results(x_values =  x1 , 
                 y_values = y_val , 
                 x_label = x_label, 
                 y_label = y_label , 
                 plot_more_than_one_fig = True)
    '''
def plot_some_plots(x_label, y_label, x_values, y_values, y_plot_labels, main_title, x_tick_values):
    
    plt.xlabel(x_label , fontsize = 14, fontname = 'Times New Roman')
    plt.ylabel(y_label, fontsize = 14, fontname = 'Times New Roman')
    plt.suptitle(main_title, fontsize = 14, fontname = 'Times New Roman', fontweight='bold')
        
    plt.xticks(fontsize = 12, fontname = "Times New Roman")
    plt.yticks(fontsize = 12, fontname = "Times New Roman")

    list_of_type_faces = ['bx-', 'go-' , 'r+-', 'b*--', 'gx--' , 'ro--']
    number_of_type_faces = len(list_of_type_faces)
    for number_of_plot in range(len(y_values)):
        plt.plot(x_values, y_values[number_of_plot], 
                list_of_type_faces[number_of_plot % number_of_type_faces], 
                label = y_plot_labels[number_of_plot], linewidth = 2)
        
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=12)
    #plt.xlim(1,5)
    plt.xticks(x_tick_values)#range(0,1001,100))#x_values)
    plt.legend(prop=font, loc = 'center')#'lower left')#fontsize = 12, fontname = "Times New Roman")'center'
    plt.show()
    
def create_plot_for_different_values_of_hidden_state_in_autoencoder():
    hidden_states = list(range(10,111,10))
    train_acc = [0.66, .76, .79, .79, .78,.78, .79, .8, .84, .83, .82]
    val_acc = [.53,.69, .68, .68, .72,.64, .64, .73, .8, .7, .77]
    plot_some_plots("#Hidden states" , "accuracy" , hidden_states, [train_acc, val_acc] , ["train acc" , "val acc"], "Twor2009" + " (delta={})".format(0.1) + ", seq_len={}".format(8))
    

def plot_stacking_paper_results():
    
    Twor2009_stacking_BoE = [46.90 , 65.7, 92.4, 64.3, 83.89, 87.3, 89.0, 88.0, 88.6, 89.0, 88.1, 88.0, 84.7, 87.0, 91.2, 89.7, 88.4, 89.60, 93.2, 86.9]
    Twor2009_stacking_SoE = [88.69, 85.1, 82.0, 79.9, 78.7, 75.8, 75.7, 73.8, 69.89, 68.2, 68.2, 65.60, 60.8, 57.4, 56.99, 55.1, 53.1, 51.6, 50.8, 46.7]
    Twor2009_stacking = [88.63, 84.96, 93.23, 79.91, 91.1, 88.99, 90.47, 89.93, 88.82, 90.72, 88.05, 89.94, 85.97, 87.57, 91.93, 90.92, 87.8, 91.45, 95.0, 86.78]
    
    Tulum2009_stacking_BoE = [96.1, 94.2, 92.6, 91.15, 90.59, 89.70, 88.96, 87.89, 86.70, 85.70, 85.10, 83.70, 81.09, 78.80, 77.3, 76.59, 71.60, 69.20, 65.70, 62.70]
    Tulum2009_stacking_SoE = [91.19, 92.59, 92.00, 90.90, 89.59, 89.20, 88.59, 88.10, 86.70, 85.39, 85.79, 84.50, 82.70, 80.70, 78.80, 77.10, 75.49, 74.00, 71.70, 69.30]  
    Tulum2009_stacking = [96.05, 94.28, 93.16, 91.86, 90.63, 90.32, 89.35, 88.00, 87.65, 85.99, 85.65, 84.90, 83.03, 80.75, 78.80, 77.50, 76.25, 74.04, 71.70, 69.10] 
    

    Tulum2010_stacking_BoE = [72.0, 70.0, 70.0, 69.0, 69.0, 69.0, 69.0, 69.0, 68.0, 68.0, 68.0, 67.0, 67.0, 65.0, 64.0, 65.0, 66.0, 64.0, 62.0, 59.0]
    Tulum2010_stacking_SoE = [76.0, 79.0, 80.0, 81.0, 82.0, 83.0, 83.0, 84.0, 85.0, 86.0, 86.0, 86.0, 88.0, 87.0, 85.0, 85.0, 82.0, 81.0, 78.0, 78.0]
    Tulum2010_stacking = [76.0, 78.0, 80.0, 81.0, 82.0, 83.0, 83.0, 84.0, 85.0, 86.0, 86.0, 86.0, 88.0, 87.0, 85.0, 84.0, 82.0, 80.0, 78.0, 78.0]
    
    x_label = '∆t'
    y_label = 'F-measure'
    y_plot_labels = ["Algorithm 1","Algorithm 2" , "Algorithm 3"]

	#y_plot_labels = ["Bag of Events" , "Sequecne of Events" , "Stacking"]

    #Twor2009
    plot_some_plots(x_label = x_label, y_label = y_label, x_values = x1, 
                   y_values = [Twor2009_stacking_BoE, Twor2009_stacking_SoE, Twor2009_stacking], 
                   y_plot_labels = y_plot_labels, 
                   main_title = "Twor2009")
    
    #Tulum2009
    plot_some_plots(x_label = x_label, y_label = y_label, x_values = x1, 
                   y_values = [Tulum2009_stacking_BoE, Tulum2009_stacking_SoE, Tulum2009_stacking], 
                   y_plot_labels = y_plot_labels, 
                   main_title = "Tulum2009")
    
    #Tulum2010
    plot_some_plots(x_label = x_label, y_label = y_label, x_values = x1, 
                   y_values = [Tulum2010_stacking_BoE, Tulum2010_stacking_SoE, Tulum2010_stacking], 
                   y_plot_labels = y_plot_labels, 
                   main_title = "Tulum2010")

def plot_LSTM_best_acc_for_each_strategy_based_on_layers():
      
    best_TBWF_acc = [max(Twor2009_TBWF_LSTM_1layer_val_acc), max(Twor2009_TBWF_LSTM_2layer_val_acc), 
	                 max(Twor2009_TBWF_LSTM_3layer_val_acc), max(Twor2009_TBWF_LSTM_4layer_val_acc)]
					 
    best_ABWF_acc = [Twor2009_ABWF_LSTM_1layer_val_acc[0], Twor2009_ABWF_LSTM_2layer_val_acc[0],
	                 Twor2009_ABWF_LSTM_3layer_val_acc[0], Twor2009_ABWF_LSTM_4layer_val_acc[0]]
					 
    best_HWF_acc =  [max(Twor2009_HWF_LSTM_1layer_val_acc), max(Twor2009_HWF_LSTM_2layer_val_acc), 
	                 max(Twor2009_HWF_LSTM_3layer_val_acc), max(Twor2009_HWF_LSTM_4layer_val_acc)]
	
    print("best_TBWF_acc:", best_TBWF_acc)
    print("best_ABWF_acc:", best_ABWF_acc)
    print("best_HWF_acc:", best_HWF_acc)
	
    plot_some_plots(x_label = "Number of Layers", 
	                y_label = "Accuracy", 
					x_values = layer_numbers, 
					y_values = [best_TBWF_acc, best_ABWF_acc, best_HWF_acc], 
					y_plot_labels = ["TBWF", "ABWF", "HWF"], 
					main_title = "Accuracy of Different Feature Engineering Strategies based on Different Number of Layers",
					x_tick_values = layer_numbers)
 

def plot_best_acc_for_each_strategy_based_on_layers_using_eval(dataset_name, LSTM_or_RNN, acc_or_f1, shuffle):
    '''
    Parameters:
	dataset_name
	LSTM_or_RNN: specify 'RNN' or 'LSTM'
	acc_or_f1: specify 'acc' or 'f1'
    '''
    sh = ''
    if shuffle:
        sh = '_with_shuffle'
    best_TBWF_acc = [max(eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1 + sh)), max(eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1+ sh)), 
	                 max(eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1+ sh)), max(eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1+ sh))]
					 
    best_ABWF_acc = [eval(dataset_name + '_ABWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1+ sh)[0], eval(dataset_name + '_ABWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1+ sh)[0], 
	                 eval(dataset_name + '_ABWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1+ sh)[0], eval(dataset_name + '_ABWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1+ sh)[0]]
					 
    best_HWF_acc =  [max(eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1+ sh)), max(eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1+ sh)), 
	                 max(eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1+ sh)), max(eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1+ sh))]
					 
    print("best_TBWF_acc:", best_TBWF_acc)
    print("best_ABWF_acc:", best_ABWF_acc)
    print("best_HWF_acc:", best_HWF_acc)
	
    plot_some_plots(x_label = "Number of Layers", 
	                y_label = "F-measure", 
					x_values = layer_numbers, 
					y_values = [best_TBWF_acc, best_ABWF_acc, best_HWF_acc], 
					y_plot_labels = ["TBWF", "ABWF", "HWF"], 
					main_title = LSTM_or_RNN + " in " + dataset_name,
					x_tick_values = layer_numbers)
 
 
def plot_different_strategies_acc_based_on_different_deltas():
    x_tick_values = range(0,1001,100)
    plot_some_plots(x_label = "∆t", 
	                y_label = "Accuracy", 
					x_values = x2, 
					y_values = [Twor2009_TBWF_LSTM_1layer_val_acc, Twor2009_TBWF_LSTM_2layer_val_acc, 
					            Twor2009_TBWF_LSTM_3layer_val_acc, Twor2009_TBWF_LSTM_4layer_val_acc],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "TBWF",
					x_tick_values = x_tick_values)
 
    plot_some_plots(x_label = "∆t", 
	                y_label = "Accuracy", 
					x_values = x2, 
					y_values = [Twor2009_HWF_LSTM_1layer_val_acc, Twor2009_HWF_LSTM_2layer_val_acc, 
					            Twor2009_HWF_LSTM_3layer_val_acc, Twor2009_HWF_LSTM_4layer_val_acc],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "HWF",
					x_tick_values = x_tick_values)

def plot_different_strategies_acc_based_on_different_deltas_using_eval(dataset_name, LSTM_or_RNN, acc_or_f1, shuffle):
    '''
    Parameters:
	dataset_name
	LSTM_or_RNN: specify 'RNN' or 'LSTM'
	acc_or_f1: specify 'acc' or 'f1'
	shuffle: if the name must have '_with_shuffle' set it to True
    '''
    sh = ''
    if shuffle:
        sh = '_with_shuffle'
		
    y_label = "F-measure"
    x_tick_values = range(0,1001, 100)
    plot_some_plots(x_label = "∆t", 
	                y_label = y_label, 
					x_values = x2, 
					y_values = [eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1 + sh), eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1 + sh), 
					            eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1 + sh), eval(dataset_name + '_TBWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1 + sh)],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "TBWF",
					x_tick_values = x_tick_values)
 
    plot_some_plots(x_label = "∆t", 
	                y_label = y_label, 
					x_values = x2, 
					y_values = [eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1 + sh), eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1 + sh), 
					            eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1 + sh), eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1 + sh)],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "HWF",
					x_tick_values = x_tick_values[0:15])
 
 
if __name__ == "__main__":
    '''
    plot_six_plots(x1, 
                   Twor2009_BOE_TBWF,
                   Twor2009_BOE_ABWF,
                   Twor2009_BOE_HWF, 
                   Twor2009_SOE_TBWF,
                   Twor2009_SOE_ABWF,
                   Twor2009_SOE_HWF)

    
    plot_six_plots(x1, 
                   Tulum2009_BOE_TBWF,
                   Tulum2009_BOE_ABWF,
                   Tulum2009_BOE_HWF, 
                   Tulum2009_SOE_TBWF,
                   Tulum2009_SOE_ABWF,
                   Tulum2009_SOE_HWF)
    
    plot_six_plots(x1, 
                   Tulum2010_BOE_TBWF,
                   Tulum2010_BOE_ABWF,
                   Tulum2010_BOE_HWF, 
                   Tulum2010_SOE_TBWF,
                   Tulum2010_SOE_ABWF,
                   Tulum2010_SOE_HWF)
    '''              
    #plot_some_plots("x" , "y", [1,2,3] , [[1,2,3], [3,4,5]], ["y1","y2"],"title", True )
    #plot_some_plots("x" , "y", [1,2,3] , [[1,2,3], [3,4,5]], ["y1","y2"],"title",  False )
    #create_plot_for_different_values_of_hidden_state_in_autoencoder()
    
    #plot_stacking_paper_results()
    #plot_LSTM_best_acc_for_each_strategy_based_on_layers()
    #plot_LSTM_different_strategies_acc_based_on_different_deltas()
    
	
    plot_best_acc_for_each_strategy_based_on_layers_using_eval(dataset_name = "Tulum2009", LSTM_or_RNN="RNN", acc_or_f1 = "acc", shuffle = True)
    
	
	#plot_different_strategies_acc_based_on_different_deltas_using_eval()
    '''
	dataset_name = "Twor2009"
    LSTM_or_RNN = "LSTM"
    acc_or_f1 = "f1"
    y_values = [eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_1layer_val_' + acc_or_f1), eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_2layer_val_' + acc_or_f1), 
					            eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_3layer_val_' + acc_or_f1), eval(dataset_name + '_HWF_' + LSTM_or_RNN + '_4layer_val_' + acc_or_f1)],
    '''
    #plot_different_strategies_acc_based_on_different_deltas_using_eval(dataset_name = "Tulum2010", LSTM_or_RNN = "RNN", acc_or_f1 = "acc", shuffle = True)

