'''
Created on Sep 26, 2018

@author: Adele
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from plot_results import plot_results

x1 = [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]
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
Tulum2010_BOE_ABWF = [90] * len(x1)
Tulum2010_BOE_HWF =  [99,99,99,98,98,93,92,92,90,88,88,88,86,86,80,80,80,80,80,80]

Tulum2010_SOE_TBWF = [76,78,80,80,80, 82,82,83,84,84,84,86,86,84,84,81,80,79,77,75]
Tulum2010_SOE_ABWF = [99] * len(x1)
Tulum2010_SOE_HWF = [84,84,84,85,88,85,85,84,88,85,85,88,88,88,88,88,86,90,92,90]



def plot_six_plots(x1, BOE_TBWF,BOE_ABWF,BOE_HWF, SOE_TBWF,SOE_ABWF,SOE_HWF):
    
    x_label = '∆t', 
    y_label = 'F-measure' , 
    plt.xlabel('∆t' , fontsize = 12, fontname = 'Times New Roman')
    plt.ylabel('F-measure', fontsize = 12, fontname = 'Times New Roman')
        
    plt.xticks(fontsize = 12, fontname = "Times New Roman")
    plt.yticks(fontsize = 12, fontname = "Times New Roman")
       
    plt.plot(x1, BOE_TBWF, 'bx-', label = 'BOE+TBWF', linewidth = 2)
    plt.plot(x1, BOE_ABWF, 'go-', label = 'BOE+ABWF', linewidth = 2)
    plt.plot(x1,BOE_HWF, 'r+-' , label = 'BOE+HWF',linewidth = 2)
    
    plt.plot(x1, SOE_TBWF, 'b*--', label = 'SOE+TBWF', linewidth = 2)
    plt.plot(x1, SOE_ABWF, 'gx--', label = 'SOE+ABWF', linewidth = 2)
    plt.plot(x1,SOE_HWF, 'ro--' , label = 'SOE+HWF',linewidth = 2)
    
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=12)
    plt.legend(prop=font, loc = 'lower left')#fontsize = 12, fontname = "Times New Roman")
    plt.show()
    '''
    y_val = [BOE_TBWF,BOE_ABWF,BOE_HWF]
    
    plot_results(x_values =  x1 , 
                 y_values = y_val , 
                 x_label = x_label, 
                 y_label = y_label , 
                 plot_more_than_one_fig = True)
    '''

if __name__ == "__main__":
    '''
    plot_six_plots(x1, 
                   Twor2009_BOE_TBWF,
                   Twor2009_BOE_ABWF,
                   Twor2009_BOE_HWF, 
                   Twor2009_SOE_TBWF,
                   Twor2009_SOE_ABWF,
                   Twor2009_SOE_HWF)

'
    plot_six_plots(x1, 
                   Tulum2009_BOE_TBWF,
                   Tulum2009_BOE_ABWF,
                   Tulum2009_BOE_HWF, 
                   Tulum2009_SOE_TBWF,
                   Tulum2009_SOE_ABWF,
                   Tulum2009_SOE_HWF)
    '''
    plot_six_plots(x1, 
                   Tulum2010_BOE_TBWF,
                   Tulum2010_BOE_ABWF,
                   Tulum2010_BOE_HWF, 
                   Tulum2010_SOE_TBWF,
                   Tulum2010_SOE_ABWF,
                   Tulum2010_SOE_HWF)