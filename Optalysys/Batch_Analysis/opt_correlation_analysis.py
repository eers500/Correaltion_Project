#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:59:50 2019

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import functions as f
from skimage.feature import peak_local_max

#%%
camera_photo = scipy.io.loadmat('camera_photo.mat')

_, _, _,camera_photo = camera_photo.values()

input_image_number = scipy.io.loadmat('input_image_number.mat')
_, _, _,input_image_number = input_image_number.values()

filter_image_number = scipy.io.loadmat('filter_image_number.mat')
_, _, _,filter_image_number = filter_image_number.values()

#%%
PKS = peak_local_max(camera_photo[:, :, 0], min_distance=1, threshold_abs=10)

plt.imshow(camera_photo[:, :, 0])
plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', facecolors='none', s=80, edgecolors='r')
plt.show()

#%%
# import plotly.graph_objects as go
# from plotly.offline import plot
#
# fig = go.Figure(data=[go.Surface(z=camera_photo[:, :, 1])])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='correlation')
# fig.show()
# plot(fig)

#%%
# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure()
ax = Axes3D(fig)

X = np.arange(1220)
Y = np.arange(1644)
X, Y = np.meshgrid(Y, X)

ax.plot_surface(X, Y, camera_photo[:, :, 0])
ax.tick_params(axis='both', labelsize=10)
ax.set_title('Cells Positions in 3D', fontsize='20')
ax.set_xlabel('x (pixels)', fontsize='18')
ax.set_ylabel('y (pixels)', fontsize='18')
ax.set_zlabel('z (slices)', fontsize='18')
pyplot.show()

#%%
A = pd.DataFrame()
