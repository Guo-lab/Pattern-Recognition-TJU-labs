from matplotlib import pyplot as plot
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

figure = plot.figure()

axes = Axes3D(figure)

X = np.array([0.01, 0.1, 1, 10, 100])
Y = np.array([10, 20, 30, 40, 50])
X,Y = np.meshgrid(X,Y)

Z = np.array([[0,	0,	0.134,	0.408,	0.51],
    [0,	0,	0.196,	0.448,	0.513],
    [0,	0,	0.206,	0.448,	0.507],
    [0,	0,	0.206,	0.431,	0.484],
    [0,	0,	0.203,	0.431,	0.471]])

surf=axes.plot_surface(X,Y,Z, cmap=plot.cm.cool, label="poly curve")

Z = np.array([[0,	0,	0.101,	0.493,	0.51],
    [0,	0,	0.267,	0.529,	0.529],
    [0,	0,	0.17,	0.553,	0.553],
    [0,	0,	0.176,	0.523,	0.523],
    [0,	0,	0.186,	0.526,	0.526]])

surf=axes.plot_surface(X,Y,Z, label="rbf curve")

Z = np.array([[0,	0,	0,	0.092,	0.095],
    [0,	0,	0.003,	0.154,	0.173],
    [0,	0,	0.003,	0.229,	0.242],
    [0,	0,	0.007,	0.268,	0.281],
    [0,	0,	0.007,	0.265,	0.275]])

surf=axes.plot_surface(X,Y,Z,cmap=plot.cm.hot, label="sigmoid curve")



axes.set_xlabel('Parameter C')
axes.set_ylabel('n_components')
axes.set_zlabel('Prediction Acc')
axes.grid(True)

'''
ax1=plot.gca()
ax1.patch.set_facecolor("white")             
ax1.patch.set_alpha(0.1)   
'''
plot.show()