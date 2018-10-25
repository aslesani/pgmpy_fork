'''
Created on Sep 26, 2018

@author: Adele
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from Validation import plot_results

x1 = [15,30,45,60,75,90,100,120,150,180,200,240,300,400,500,600,700,800,900,1000]
BOE_TBWF = [80,80,92,90,87,91,90,88,88,86,89,88,83,89,91,92,89,93,96,86]
BOE_ABWF = [73] * len(x1)
BOE_HWF = [70,75,77,76,76,77,76,76,75,74,75,76,75,73,73,74,74,73,74,73]

SOE_TBWF = [87,80,81,76,75,73,75,72,67,65,66,63,58,45,55,53,52,46,50,47]
SOE_ABWF = [87] * len(x1)
SOE_HWF = [89,88,88,87,87,88,87,87,87,86,87,86,87,87,86,86,87,86,86,86]

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
#plt.legend(prop=font, loc = 'lower left')#fontsize = 12, fontname = "Times New Roman")
y_val = [BOE_TBWF,BOE_ABWF,BOE_HWF]

'''
plot_results(x_values =  x1 , 
             y_values = y_val , 
             x_label = x_label, 
             y_label = y_label , 
             plot_more_than_one_fig = True)
 '''