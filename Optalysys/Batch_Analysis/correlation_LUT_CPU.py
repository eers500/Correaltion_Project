#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:50:28 2019

@author: erick
"""

import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
import time
from scipy import ndimage
from numba import vectorize, jit

#%%
# Import Video correlate
# VID = f.videoImport("E://PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
# VID = f.videoImport("/home/erick/Documents/PhD/23_10_19/0-300_10x_100Hz_45um_frame_stack_every10um.avi", 0).astype('uint8')
# VID = VID[:, :, :21]
VID = f.videoImport('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_frame_stack_20um-21_frames.avi', 0)
MAX_VID = np.max(VID)
VID = np.uint8(255*(VID / MAX_VID))

#%%
# Import LUT form images
#LUT = [cv2.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = [mpimg.imread(file) for file in np.sort(glob.glob("E://PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
# LUT = np.swapaxes(np.swapaxes(LUT, 0, 1), 1, 2)
# LUT = f.videoImport('/home/erick/Documents/PhD/Colloids/20x_50Hz_100us_642nm_colloids_frame_stack_20um-particle_1.avi', 0)
# LUT = np.uint8(255*(LUT / MAX_VID))

# LUT = VID[151-18:151+18, 162-18:162+18, :]  # Particle 1 for collodids
# LUT = VID[77-18:77+18, 332-18:332+18, :]  # P2
LUT = VID[379-18:379+18, 130-18:130+18, :]  # P3
# LUT = VID[369-18:369+18, 292-18:292+18, :]  # P4

#%% 
# # Plot raw frames
# plt.subplot(3, 2, 1); plt.imshow(VID[:, :, 0], cmap='gray')
# plt.subplot(3, 2, 2); plt.imshow(LUT[:, :, 0], cmap='gray')

# plt.subplot(3, 2, 3); plt.imshow(VID[:, :, 10], cmap='gray')
# plt.subplot(3, 2, 4); plt.imshow(LUT[:, :, 10], cmap='gray')

# plt.subplot(3, 2, 5); plt.imshow(VID[:, :, 20], cmap='gray')
# plt.subplot(3, 2, 6); plt.imshow(LUT[:, :, 20], cmap='gray')

# # Plot binary frames

# plt.subplot(3, 2, 1); plt.imshow(VID_BINARY[:, :, 0], cmap='gray')
# plt.subplot(3, 2, 2); plt.imshow(LUT_BINARY[:, :, 0], cmap='gray')

# plt.subplot(3, 2, 3); plt.imshow(VID_BINARY[:, :, 10], cmap='gray')
# plt.subplot(3, 2, 4); plt.imshow(LUT_BINARY[:, :, 10], cmap='gray')

# plt.subplot(3, 2, 5); plt.imshow(VID_BINARY[:, :, 20], cmap='gray')
# plt.subplot(3, 2, 6); plt.imshow(LUT_BINARY[:, :, 20], cmap='gray')


#%%
# Compute Correlation for all filters
LUT_BINARY = np.zeros(np.shape(LUT))
VID_BINARY = np.zeros(np.shape(VID))

LUT_BINARY[LUT >= np.mean(LUT)] = 255
VID_BINARY[VID >= np.mean(LUT)] = 255

CORR = np.empty((np.shape(VID)[0], np.shape(VID)[1] , np.shape(VID)[2] * np.shape(LUT)[2]), dtype='float32')

A = np.repeat(VID_BINARY, repeats=21, axis=-1).astype('float32')
B = np.tile(LUT_BINARY, 21).astype('float32')

#%%
# Correltion in GPU
# @vectorize(["complex128(complex128, complex128)"], target='cuda')
@jit(nopython=True) 
def corr_gpu(a, b):
    return a * np.conj(b);

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

BB = np.empty_like(A)
for k in range(np.shape(B)[2]):
    BB[:, :, k] = np.pad(B[:, :, k], 238)

FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

C = np.empty_like(A)
T0 = time.time()
for k in range(np.shape(A)[2]):
    print(k)
    R = corr_gpu(FT(A[:, :, k]), FT(BB[:, :, k]))
    C[:, :, k] = np.abs(IFT(R))
T = time.time()- T0
print(T)

# 22 seconds

# plt.imshow(np.abs(C))
# plt.colorbar()
#%%
# Calculate correlation in CPU
T0 = time.time()
for k in range(np.shape(A)[2]):
    print(k)
    CORR[:, :, k] = ndimage.filters.correlate(A[:, :, k], B[:, :, k], mode='wrap')
T = time.time()- T0
print(T/60)

# ~ 3 min

# Convert to 8-bit
#C = np.empty_like(CORR)
#for i in range(np.shape(CORR)[2]):
#    C[:, :, i] = 255 * (CORR[:, :, i] / np.max(CORR[:, :, i]))
#C = np.uint8(C)
# Optical Power normalization
#for k in range(np.shape(CORR)[2]):
#    CORR[:, :, k] = CORR[:, :, k] / np.sum(CORR[:, :, k]**2)

#%%
# Import correlation result saves from previous step
# CORR = np.load('CORR_CPU.npy')
CORR = np.load('CORR_CPU_Colloids_P1.npy')
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
# # CORR = C
# MAX_MMM = np.empty(21)
# for i in range(21):
#     k = i*21
#     T = CORR[:, :, k:k+21]
    
#     MM = np.empty(21)
#     for kk in range(21):
#         MM[kk] = np.max(T[:, :, kk]) 
    
#     MAX_MM = np.where(MM == np.max(MM))
#     MAX_MMM[i] = MAX_MM[0][0]
#     print(MAX_MMM)
    
# #     plt.plot(np.arange(21), MM, 'o-' )
# #     plt.xticks(ticks=np.arange(21))
# #     plt.title(np.str(MAX_MM[0][0])) 

# # plt.legend(np.arange(21))    
# # plt.grid()

# # CO = np.empty((512, 512, 21))
# # for k in range(21):
# #     CO[:, :, k] = CORR[:, :, k*22]
    
#%%
  
    
#%%
# Plot images with coordinates of maximum correlation    
k = 20       
plt.imshow(VID[:, :, k], cmap='gray')
plt.scatter(LOCS[k*21+k,1], LOCS[k*21+k, 0], marker='o', color='r', facecolors='none')
plt.show()

#%%
# Check system "ideal" filter-image combination
# Not all combination have good correlation image
LL = np.empty((21, 2))
for i in range(21):
#i = 20
    k = 22*i
    # plt.figure(i+1)
    # plt.imshow(CORR[:, :, k])
    L = np.where(CORR[:, :, k] == np.max(CORR[:, :, k]))
    LOCS0, LOCS1 = L[0][0], L[1][0]
    LL[i, 0], LL[i, 1] = LOCS0, LOCS1
    # plt.scatter(LOCS1, LOCS0, marker='o', color='r', facecolors='none')
    # plt.title('CPU: '+np.str(i+1))
    
    plt.show()

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
