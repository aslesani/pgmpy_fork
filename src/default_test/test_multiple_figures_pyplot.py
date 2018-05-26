'''
Created on April 27, 2018

@author: Adele
'''
# Working with multiple figure windows and subplots
import matplotlib.pyplot as plt
import numpy as np

from Validation import plot_results


def plot_multiple_figures():
    t = np.arange(0.0, 2.0, 0.01)
    s1 = np.sin(2*np.pi*t)
    s2 = np.sin(4*np.pi*t)
    s3 = np.sin(8*np.pi*t)
    s4 = np.sin(16*np.pi*t)
    
    
    
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, s1 , 's')
    #by adele
    plt.plot(t, s2 , 's')
    #plt.plot(t, s3)
    #plt.plot(t, s4)
    
    
    
    '''
    plt.subplot(212)
    plt.plot(t, 2*s1)
    
    plt.figure(2)
    plt.plot(t, s2)
    '''
    '''
    # now switch back to figure 1 and make some changes
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, s2, 's')
    ax = plt.gca()
    ax.set_xticklabels([])
    '''
    plt.show()
    
    
def different_format_styles():
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)
    
    # red dashes, blue squares and green triangles
    # then filled circle with connecting line
    plt.clf()
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.plot(t, t+60, 'o', linestyle='-', color='g')


if __name__ == "__main__":
    x_val = [1,2,3,4]
    y_val = [1,4,9,16]
    y_val2 = [5,5,5,16]
    
    different_format_styles()
    #plot_results(x_val , [y_val , y_val2] , x_label='' , y_label='' , plot_more_than_one_fig=True)
