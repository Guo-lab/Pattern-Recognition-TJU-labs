from matplotlib import pyplot as plot
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

figure = plot.figure()

axes = Axes3D(figure)

X = np.array([0.01, 0.1, 1, 10, 100])
Y = np.array([20, 40, 60, 80, 95])
X,Y = np.meshgrid(X,Y)

Z = np.array([[0,	0,	0,	0,	0],
    [0,	0,	0.013,	0.016,	0.02],
    [0,	0,	0.092,	0.34,	0.461],
    [0,	0,	0.209,	0.454,	0.51],
    [0,	0,	0.203,	0.395,	0.425]])

surf=axes.plot_surface(X,Y,Z, cmap=plot.cm.cool, label="poly curve")

Z = np.array([[0,	0,	0,	0,	0],
    [0,	0,	0,	0.033,	0.078],
    [0,	0,	0.085,	0.435,	0.493],
    [0,	0,	0.17,	0.529,	0.529],
    [0,	0,	0.18,	0.493,	0.493]])

surf=axes.plot_surface(X,Y,Z, cmap=plot.cm.hot,label="rbf curve")

Z = np.array([[0,	0,	0,	0.007,	0.003],
    [0,	0,	0,	0,	0.1],
    [0,	0,	0,	0.052,	0.088],
    [0,	0,	0.003,	0.199,	0.222],
    [0,	0,	0.013,	0.32,	0.382]])

surf=axes.plot_surface(X,Y,Z, label="sigmoid curve")



axes.set_xlabel('Parameter C')
axes.set_ylabel('n_components (%)')
axes.set_zlabel('Prediction Acc')
axes.grid(True)

'''
ax1=plot.gca()
ax1.patch.set_facecolor("white")             
ax1.patch.set_alpha(0.1)   
'''
plot.show()