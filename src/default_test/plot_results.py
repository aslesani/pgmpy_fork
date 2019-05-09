import matplotlib.pyplot as plt

def plot_results(x_values , y_values, x_label, y_label , plot_more_than_one_fig = False):
    '''
    plot the figure
    
    Parameters:
    ===========
    x_values: the list of x values
    y_values: the list of y values
    x_label:
    y_label:
    plot_more_than_one_fig: if True, plot more than one fig and y_vlaues is a list of values
                            so that each values is as long as x_values
                            if is False, the y_label is just a list of corrsponding values of x_values
    
    '''
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    #plt.axes([.2, .2, .7, .7])
    if plot_more_than_one_fig:
        for y_val in y_values:
            plt.plot(x_values, y_val)#, linewidth=1)

    else:
        plt.plot(x_values, y_values)
        
    plt.xlabel(x_label , fontsize = 12, fontname = 'Times New Roman')
    plt.ylabel(y_label, fontsize = 12, fontname = 'Times New Roman')
    
    plt.xticks(fontsize = 12, fontname = "Times New Roman")
    plt.yticks(fontsize = 12, fontname = "Times New Roman")
   
    plt.show()

