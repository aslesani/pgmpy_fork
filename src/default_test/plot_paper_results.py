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
def plot_some_plots(x_label, y_label, x_values, y_values, y_plot_labels, main_title):
    
    plt.xlabel(x_label , fontsize = 14, fontname = 'Times New Roman')
    plt.ylabel(y_label, fontsize = 14, fontname = 'Times New Roman')
    plt.suptitle(main_title, fontsize = 14, fontname = 'Times New Roman', fontweight='bold')
        
    plt.xticks(fontsize = 14, fontname = "Times New Roman")
    plt.yticks(fontsize = 14, fontname = "Times New Roman")

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
    plt.xticks(x_values)
    plt.legend(prop=font, loc = 'lower left')#fontsize = 12, fontname = "Times New Roman")
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
					main_title = "Accuracy of Different Feature Engineering Strategies based on Different Number of Layers")
 
 
def plot_LSTM_different_strategies_acc_based_on_different_deltas():
 
    plot_some_plots(x_label = "∆t", 
	                y_label = "Accuracy", 
					x_values = x2, 
					y_values = [Twor2009_TBWF_LSTM_1layer_val_acc, Twor2009_TBWF_LSTM_2layer_val_acc, 
					            Twor2009_TBWF_LSTM_3layer_val_acc, Twor2009_TBWF_LSTM_4layer_val_acc],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "TBWF")
 
    plot_some_plots(x_label = "∆t", 
	                y_label = "Accuracy", 
					x_values = x2, 
					y_values = [Twor2009_HWF_LSTM_1layer_val_acc, Twor2009_HWF_LSTM_2layer_val_acc, 
					            Twor2009_HWF_LSTM_3layer_val_acc, Twor2009_HWF_LSTM_4layer_val_acc],
					y_plot_labels = ["1 Layer", "2 Layers", "3 Layers", "4 Layers"], 
					main_title = "HWF")
 
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
    plot_LSTM_best_acc_for_each_strategy_based_on_layers()
    #plot_LSTM_different_strategies_acc_based_on_different_deltas()