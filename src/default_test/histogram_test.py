'''
Created on Feb 17, 2018

@author: cloud
'''

import matplotlib.pyplot as plt
import numpy as np
#% matplotlib inline
x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability')

x = np.arange(3)
plt.bar(x, height= [1,2,3])
plt.xticks(x+.5, ['a','b','c']);


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)


bins = range(1,11)
plt.plot(bins)#, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

#plt.show()


import matplotlib as mpl
import matplotlib.pyplot as plt

xedges = [0, 1, 1.5, 3, 5]
yedges = [0, 2, 3, 4, 6]

x = np.random.normal(3, 1, 100)
y = np.random.normal(1, 1, 100)
H, xedges, yedges = np.histogram2d(y, x, bins=(xedges, yedges))
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131)
ax.set_title('imshow: equidistant')
im = plt.imshow(H, interpolation='nearest', origin='low',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


import pandas as pd
data = [[1,2,3] , [1,2,3] , [4,5,6]]
a = pd.DataFrame( data, columns = ['c1' , 'c2' , 'c3'] )
print(type(a.groupby(['c1', 'c2']).agg(['count'])))



#b = a['c3'].value_counts()#.plot(kind='barh')
#a.set_index('c3')['c2'].plot.bar()

