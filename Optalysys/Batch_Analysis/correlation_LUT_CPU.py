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
LUT = [mpimg.imread(file) for file in np.sort(glob.glob("E://PhD/23_10_19/LUT_MANUAL/*.png"))]
LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)
LUT = 255*(LUT / np.max(LUT))
#%%
# Import Video correlate
VID = f.videoImport("E://PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')

#%%

LUT_BINARY = np.zeros(np.shape(LUT))
VID_BINARY = np.zeros(np.shape(VID))

LUT_BINARY[LUT >= np.mean(LUT)] = 255
VID_BINARY[VID >= np.mean(VID)] = 255

CORR = np.empty((np.shape(VID)[0], np.shape(VID)[1] , np.shape(VID)[2] * np.shape(LUT)[2]))

#for i in range(np.shape(VID)[2]):
#    for j in range(np.shape(LUT)[2]):
#        print((i, j))
#        CORR[:, :, 21*i+j] = ndimage.filters.correlate(VID_BINARY[:, :, i], LUT_BINARY[:, :, j], mode='wrap')
#        print(20*i + j)
#
#MAX = []
#for k in range(651):
#    MAX.append(np.max(CORR[:, :, k]))

A = np.repeat(VID_BINARY, repeats=21, axis=-1)
B = np.tile(LUT_BINARY, 31)

VID_FT = np.fft.fftshift(np.fft.fft2(A))
LUT_FT = np.fft.fftshift(np.fft.fftn(B, s=(512, 510, 651)))

R = VID_FT * np.conj(LUT_FT)
CORR = np.real(np.fft.ifftshift(np.fft.ifft2(R)))
#%%
AA = VID_BINARY[:, :, 0]
BB = LUT_BINARY[:, :, 0]
BB_FT = np.fft.fftshift(np.fft.fft2(BB))

C = ndimage.filters.correlate(AA, BB)
#%%
 plt.imshow(B, cmap='gray')
 plt.show()

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