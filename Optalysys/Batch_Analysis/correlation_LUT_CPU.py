#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:50:28 2019

@author: erick
"""

import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
import time
from scipy import ndimage

#%%
# Import LUT form images
# LUT = [cv2.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = [mpimg.imread(file) for file in np.sort(glob.glob("E://PhD/23_10_19/LUT_MANUAL/*.png"))]
#LUT = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)
LUT = np.uint8(255*(LUT / np.max(LUT)))
#%%
# Import Video correlate
VID = f.videoImport("E://PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
#VID = f.videoImport("/home/erick/Documents/PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
VID = VID[:, :, :21]

#%%
#plt.subplot(3, 2, 1); plt.imshow(VID[:, :, 0], cmap='gray')
#plt.subplot(3, 2, 2); plt.imshow(LUT[:, :, 0], cmap='gray')
#
#plt.subplot(3, 2, 3); plt.imshow(VID[:, :, 10], cmap='gray')
#plt.subplot(3, 2, 4); plt.imshow(LUT[:, :, 10], cmap='gray')
#
#plt.subplot(3, 2, 5); plt.imshow(VID[:, :, 20], cmap='gray')
#plt.subplot(3, 2, 6); plt.imshow(LUT[:, :, 20], cmap='gray')


#%%
# Compute Correlation for all filters
LUT_BINARY = np.zeros(np.shape(LUT))
VID_BINARY = np.zeros(np.shape(VID))

LUT_BINARY[LUT >= np.mean(LUT)] = 255
VID_BINARY[VID >= np.mean(VID)] = 255

CORR = np.empty((np.shape(VID)[0], np.shape(VID)[1] , np.shape(VID)[2] * np.shape(LUT)[2]), dtype='float32')

A = np.repeat(VID_BINARY, repeats=21, axis=-1).astype('float32')
B = np.tile(LUT_BINARY, 31)

T0 = time.time()
for k in range(np.shape(A)[2]):
    print(k)
    CORR[:, :, k] = ndimage.filters.correlate(A[:, :, k], B[:, :, k], mode='wrap')
T = time.time()- T0
print(T/60)

# Convert to 8-bit
#C = np.empty_like(CORR)
#for i in range(np.shape(CORR)[2]):
#    C[:, :, i] = 255 * (CORR[:, :, i] / np.max(CORR[:, :, i]))
#C = np.uint8(C)
# Optical Power normalization
#for k in range(np.shape(CORR)[2]):
#    CORR[:, :, k] = CORR[:, :, k] / np.sum(CORR[:, :, k]**2)


# Get (x,y) coordinates of maximum correlation spots
MAX = []
LOCS = np.empty((np.shape(CORR)[2], 2))
for k in range(np.shape(CORR)[2]):
    MAX.append(np.max(CORR[:, :, k]))
    L = np.where(CORR[:, :, k] == np.max(CORR[:, :, k]))
    LOCS[k, 0], LOCS[k, 1] = L[0][0], L[1][0]

# Get maximum correlation filter for all images    
MAX_FILT = np.empty(np.shape(VID)[2]) 
for i in range(np.shape(VID)[2]):
    M = MAX[i*21:i*21+21]
    M = np.array(M)
    MAX_FILT[i] = np.where(np.max(M) == M)[0][0]
    
#%%
# Plot images with coordinates of maximum correlation    
#k = 15         
#plt.imshow(VID[:, :, k], cmap='gray')
#plt.scatter(LOCS[k*21+k,1], LOCS[k*21+k, 0], marker='o', color='r', facecolors='none')

#%%
# LUT_BINARY = np.zeros(np.shape(LUT), dtype='uint8')
# VID_BINARY = np.zeros(np.shape(VID), dtype='uint8')

# LUT_BINARY[LUT >= np.mean(LUT)] = 255
# VID_BINARY[VID >= np.mean(VID)] = 255

# BC = np.zeros((512, 510, 21))
# BC[:np.shape(LUT_BINARY)[0], :np.shape(LUT_BINARY)[1], :] = LUT_BINARY
# LUT_BINARY = BC
# del BC
# NI, NJ, _ = np.shape(LUT_BINARY)
# Ni, Nj, _ = np.shape(LUT)
# LUT_BINARY = np.roll(LUT_BINARY, (int(np.floor(NI/2 - Ni/2)), int(np.floor(NJ/2 - Nj/2))), axis=(0, 1))

# FT_VID = np.empty_like(VID_BINARY, dtype='complex64')
# FT_LUT = np.empty_like(LUT_BINARY, dtype='complex64')

# for k in range(np.shape(VID_BINARY)[2]):
#     FT_VID[:, :, k] = np.fft.fftshift(np.fft.fft2(VID_BINARY[:, :, k]))
    
# for k in range(np.shape(LUT_BINARY)[2]):
#     FT_LUT[:, :, k] = np.fft.fftshift(np.fft.fft2(LUT_BINARY[:, :, k]))

# A = np.repeat(FT_VID, repeats=21, axis=-1)
# B = np.tile(FT_LUT, 31)

# CORR = np.real(A*np.conj(B))
# CORR = CORR / (np.sum(VID_BINARY**2)*np.sum(LUT_BINARY**2))

#%%
# plt.subplot(1,2,1)
# plt.imshow(A, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(B, cmap='gray')

#%%
#import plotly.express as px
#import pandas as pd
#import plotly.graph_objects as go
#from plotly.offline import plot 
#
#fig = go.Figure(data=[go.Surface(z=CORR[:, :, -1])])
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
#fig.show()
#plot(fig)
