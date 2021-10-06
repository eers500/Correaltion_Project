# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 14:51:34 2021

@author: eers500
"""

import glob
import numpy as np
import matplotlib as mpl
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import functions as f
import time
import easygui as gui
from scipy import ndimage
from numba import vectorize, jit

#%% Import Video correlate
path_vid = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
VID = f.videoImport(path_vid, 0)
# MAX_VID = np.max(VID)
# VID = np.uint8(255*(VID / MAX_VID))

#%% Import LUT form images
path_lut = gui.fileopenbox(default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/')
LUT = f.videoImport(path_lut, 0)

#%% Prepare arrays
VID_zn = np.empty_like(VID)
for k in range(VID.shape[-1]):
    A = VID[:,:,k]
    VID_zn[:, :, k] = (A-np.mean(A))/np.std(A)

LUT_zn = np.empty_like(LUT)
for k in range(LUT.shape[-1]):
    A = LUT[:,:,k]
    LUT_zn[:, :, k] = (A-np.mean(A))/np.std(A)

A = np.repeat(VID_zn, repeats=LUT_zn.shape[-1], axis=-1)
B = np.tile(LUT_zn, VID_zn.shape[-1])

#%% Correltion in GPU
#@vectorize(["complex128(complex128, complex128)"], target='cuda')   #not good
@jit(nopython=True) 
def corr_gpu(a, b):
    return a*np.conj(b)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

BB = np.empty_like(A)
for k in range(np.shape(B)[2]):
    # BB[:, :, k] = np.pad(B[:, :, k], int((1024-110)/2))
    BB[:, :, k] = np.pad(B[:, :, k], int((A.shape[0]-B.shape[0])/2))
# del B
FT = lambda x: np.fft.fftshift(np.fft.fft2(x))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(X))

C = np.empty_like(A, dtype='float32')
T0 = time.time()
T = []
for k in range(np.shape(A)[2]):
# for k in range(100):
    print(k)
    BFT = FT(BB[:,:,k])
    R = corr_gpu(FT(A[:, :, k]), BFT).astype('complex64')
    # C[:, :, k] = np.abs(IFT(R))
    C[:, :, k] = np.abs(IFT(R / np.sum(BFT)))    # Normalize with sum of pixel values of filters
    T.append((time.time()-T0)/60)
print(T[-1])

# del A, BB
# CC = np.reshape(C, (226*39000, 226))
# np.savetxt('F:\PhD\Archea_LW\LUT_CES_30\GPU_corr_21.58min_226.txt', CC, fmt='%i', delimiter=',')

# np.savez_compressed('F:\PhD\Archea_LW\LUT_CES_44\GPU_corr_7.78min_226_400frames_every5_normalised_filter_sum_squared.npz', a=C)
# np.savez_compressed('F:\\PhD\\E_coli\\may2021\\5\\20x_100Hz_05us_EcoliHCB1-07_GPU_corr_275of550frames.npz', a=C)
# np.save('F:\PhD\E_coli\may2021\5\20x_100Hz_05us_EcoliHCB1-07_GPU_corr.npy', a=C)

# 22 seconds
#%%
# plt.imshow(C[:, :, 0])
# plt.colorbar()
#%% Calculate correlation in CPU
T0 = time.time()
CORR = np.empty_like(A)
for k in range(np.shape(A)[2]):
    print(k)
    CORR[:, :, k] = ndimage.filters.correlate(A[:, :, k], B[:, :, k], mode='wrap')
    print((time.time()-T0)/60)
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

#%% Import correlation result save from previous step
# CORR = np.load('CORR_CPU.npy')
# CORR = np.load('CORR_CPU_Colloids_P4.npy')
path = '/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/'

# CORR = np.load(path+'substack_GPU_Result.npy')
CORR = np.load(path+'substack_CES_GPU_Result.npy')

#%% Get (x,y) coordinates of maximum correlation spots
number = int(np.sqrt(np.shape(CORR)[2]))
number_of_images = number
number_of_filters = number

# number_of_images = np.shape(VID)[2]
# number_of_filters = np.shape(LUT)[2]

MAX = []
LOCS = np.empty((np.shape(CORR)[2], 2))
for k in range(np.shape(CORR)[2]):
    MAX.append(np.max(CORR[:, :, k]))
    L = np.where(CORR[:, :, k] == np.max(CORR[:, :, k]))
    LOCS[k, 0], LOCS[k, 1] = L[0][0], L[1][0]

# Columns are the input images and rows are the input filters
MAX = np.reshape(MAX, (number_of_filters, number_of_images), 'F')

# Get maximum correlation filter for all images    
MAX_FILT = np.empty(number_of_images) 
for i in range(number_of_images):
    MAX_FILT[i] = np.where(MAX[:, i] == MAX[:, i].max())[0][0]

plt.imshow(MAX, cmap='viridis')
plt.show()
print(MAX_FILT)
 
#%%
# import plotly.express as px
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.offline import plot 

# fig = go.Figure(data=[go.Surface(z=MAX)])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
# fig.show()
# plot(fig)
