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
CAMERA_PHOTO = scipy.io.loadmat('camera_photo.mat')
_, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

FILTERS = scipy.io.loadmat('filters.mat')
_, _, _, FILTERS = FILTERS.values()

INPUT_IMAGE_NUMBER = scipy.io.loadmat('input_image_number.mat')
_, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values()

FILTER_IMAGE_NUMBER = scipy.io.loadmat('filter_image_number.mat')
_, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()

A, _ = np.where(INPUT_IMAGE_NUMBER > 21)
CAMERA_PHOTO = np.delete(CAMERA_PHOTO, A, axis=-1)
INPUT_IMAGE_NUMBER = np.delete(INPUT_IMAGE_NUMBER, A)
FILTER_IMAGE_NUMBER = np.delete(FILTER_IMAGE_NUMBER , A)

#%%
# INDEX = 6
# PKS = peak_local_max(CAMERA_PHOTO[:, :, INDEX], min_distance=1, threshold_abs=15)
#
# plt.imshow(CAMERA_PHOTO[:, :, INDEX])
# plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', facecolors='none', s=80, edgecolors='r')
# plt.show()

#%%
# import plotly.graph_objects as go
# from plotly.offline import plot
#
# fig = go.Figure(data=[go.Surface(z=CAMERA_PHOTO[:, :, 1])])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='correlation')
# fig.show()
# plot(fig)

#%%
# 3D Scatter Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# X = np.arange(1220)
# Y = np.arange(1644)
# X, Y = np.meshgrid(Y, X)
#
# ax.plot_surface(X, Y, CAMERA_PHOTO[:, :, 0])
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
# pyplot.show()

#%%
# for j in range(np.shape(CAMERA_PHOTO)[2]):
#     CAMERA_PHOTO[:, :, j] = CAMERA_PHOTO[:, :, j] / (np.sum(np.real(FILTERS[:, :, FILTER_IMAGE_NUMBER[j]-1]))*np.sum(np.abs(CAMERA_PHOTO[:, :, j])))

#%%
# Normalization to compare images
SUM_FILTS = np.empty(np.shape(CAMERA_PHOTO)[2])
SUM_CAM = np.empty_like(SUM_FILTS)
MULT = np.empty_like(SUM_FILTS)
CAMERA_PHOTOS = np.empty_like(CAMERA_PHOTO)

# FILTERS[FILTERS <= 0] = 0
# FILTERS[FILTERS > 0] = 255


# for i in range(np.shape(CAMERA_PHOTO)[2]):
#     # SUM_FILTS[i] = np.sum(np.real(FILTERS[:, :, FILTER_IMAGE_NUMBER[i]-1]))
#     SUM_CAM[i] = np.sum(np.real(CAMERA_PHOTO[:, :, i]))
#     # MULT[i] = SUM_FILTS[i] + SUM_CAM[i]
#     # CAMERA_PHOTOS[:, :, i] = CAMERA_PHOTO[:, :, i] / MULT[i]
#     CAMERA_PHOTOS[:, :, i] = CAMERA_PHOTO[:, :, i]

SUM_CAM = np.sum(CAMERA_PHOTO, axis=(0, 1))
SUM_FILT = np.sum(FILTERS, axis=(0, 1))
CAMERA_PHOTOS = CAMERA_PHOTO / (np.max(SUM_CAM) + np.max(SUM_FILT))
CAMERA_PHOTO = CAMERA_PHOTOS

# plt.figure()
# plt.plot(SUM_FILTS)
#
# plt.figure()
# plt.plot(SUM_CAM)
#
# plt.figure()
# plt.plot(MULT)
#
# plt.figure()
# plt.imshow(CAMERA_PHOTOS[:, :, 0])


#%%
# Histogram equalization and normalization
# CAMS, cdf = f.histeq(CAMERA_PHOTO)

#%%
# For the first image (#1), we want to compare the correlation with all the filters
# The indices of image #1 in its array is:
IMAGE_FILTER_PAIR = np.zeros((len(np.unique(INPUT_IMAGE_NUMBER)), 2))

for k in range(len(np.unique(INPUT_IMAGE_NUMBER))):
    IM_NUMBER = k+1
    ID_IM = np.where(INPUT_IMAGE_NUMBER == IM_NUMBER)
    ID_IM = np.ravel(ID_IM)

    # Get max values of every filter correlation with image IM_NUMBER to compare
    MAX = np.empty(0, dtype='int8')
    # MAX_COORD = np.empty((1, 2))

    for i in range(len(ID_IM)):
        # MAX.append(np.max(CAMERA_PHOTO[:, :, ID_IM[i]]))
        # CAM_PHOTO_NORM = CAMERA_PHOTO[:, :, ID_IM[i]]
        MAX = np.append(MAX, np.max(CAMERA_PHOTO[:, :, ID_IM[i]]))
        # np.where()

    # The filter number with maximum correlation with image IM_NUMBER is
    I_MAX = np.where(MAX == np.max(MAX))

    # The filter with maximum correlation is
    MAX_CORR_FILTER = FILTER_IMAGE_NUMBER[ID_IM[I_MAX]]

    # The image, and max correlation filter pair is then
    IMAGE_FILTER_PAIR[k, 0] = IM_NUMBER
    IMAGE_FILTER_PAIR[k, 1] = MAX_CORR_FILTER[0]
    # IMAGE_FILTER_PAIR[k] = [IM_NUMBER, MAX_CORR_FILTER[0]]

print(IMAGE_FILTER_PAIR)

#%%
# IMAGE = 3
# ID1 = np.where(INPUT_IMAGE_NUMBER != IMAGE)
# PHOTOS_1 = np.delete(CAMERA_PHOTO, ID1, axis=-1)

#%%
# for i in range(len(ID_IM)):
#     plt.figure()
#     plt.imshow(CAMERA_PHOTO[:, :, ID_IM[i]])