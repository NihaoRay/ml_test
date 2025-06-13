import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib
# matplotlib inline
from matplotlib import animation
from IPython.display import HTML


from ipywidgets import interact
# workaround function for strange interact implementation
def showInInteract():
    import inspect
    for i in range(5):
        if 'interaction.py' in inspect.stack()[i][1]: plt.show()

def showSin(f):
    t = np.arange(-1, 1, 1/1000)
    plt.plot(t, np.sin(2*np.pi*f*t * t))
    plt.plot(t, np.cos(2*np.pi*f/(t**2 + 1)))
    plt.text(0, 0.5, 'f = %.1f' % f, bbox=dict(facecolor='white')) # todo 这个函数需要仔细查询与学习一下
    plt.show()

# print ("Numpy version:      ", np.version.full_version)
# print ("Scipy version:      ", scipy.version.full_version)
# print ("Matplotlib version: ", matplotlib.__version__)

# t = np.arange(-5, 5, 1/20)
#
# plt.plot(t, np.sin(t*t)/(t*t))
# plt.grid()
# plt.xlabel('$t$')
# plt.ylabel(r'$\sin(t^2)$')

showSin(5)
plt.show()