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
from scipy import ndimage

#%%
# Import LUT form images
# LUT = [cv2.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
#LUT = [mpimg.imread(file) for file in np.sort(glob.glob("E://PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)
LUT = np.uint8(255*(LUT / np.max(LUT)))
#%%
# Import Video correlate
#VID = f.videoImport("E://PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
VID = f.videoImport("/home/erick/Documents/PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')

#%%

LUT_BINARY = np.zeros(np.shape(LUT))
VID_BINARY = np.zeros(np.shape(VID))

LUT_BINARY[LUT >= np.mean(LUT)] = 255
VID_BINARY[VID >= np.mean(VID)] = 255

CORR = np.empty((np.shape(VID)[0], np.shape(VID)[1] , np.shape(VID)[2] * np.shape(LUT)[2]), dtype='float32')

A = np.repeat(VID_BINARY, repeats=21, axis=-1).astype('float32')
B = np.tile(LUT_BINARY, 31)
#B = np.zeros_like(A)
#B[:np.shape(BB)[0], :np.shape(BB)[1], :] = BB
#del BB
#NI, NJ, _ = np.shape(B)
#Ni, Nj, _ = np.shape(LUT)
#B = np.roll(B, (int(np.floor(NI/2 - Ni/2)), int(np.floor(NJ/2 - Nj/2))), axis=(0, 1)).astype('float32')

#A = np.fft.fftshift(np.fft.fft2(A, axes=(0, 1))).astype('complex64') # 31
#B = np.fft.ifftshift(np.fft.ifft2(B, axes=(0, 1))).astype('complex64')  #21

for k in range(np.shape(A)[2]):
    print(k)
    CORR[:, :, k] = ndimage.filters.correlate(A[:, :, k], B[:, :, k], mode='wrap')

#CORR = np.abs(np.fft.ifftshift(np.fft.ifft2(A * np.conj(B))))

C = np.empty_like(CORR)
for i in range(np.shape(CORR)[2]):
    C[:, :, i] = 255 * (CORR[:, :, i] / np.max(CORR[:, :, i]))

MAX = []
for k in range(651):
    MAX.append(np.max(CORR[:, :, k]))
    
MAX_FILT = np.empty(31) 
for i in range(31):
    M = MAX[i*21:i*21+21]
    M = np.array(M)
    MAX_FILT[i] = np.where(np.max(M) == M)[0][0]
    
        

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
#fig = go.Figure(data=[go.Surface(z=CORR)])
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
#fig.show()
#plot(fig)
