# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:32:39 2021

@author: eers500
"""

import numpy as np
import matplotlib as mpl
mpl.rc('figure',  figsize=(10, 6))
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import functions as f
import easygui as gui
from skimage.feature import peak_local_max
import time

mode = True   # Set false to select GPU .npy arrays

if mode: 
#%
# PATH = '/media/erick/NuevoVol/LINUX_LAP/PhD/'
    PATHS = gui.fileopenbox(msg='Select File',
                            title='Files',
                            # default='/home/erick/Documents/PhD/Correaltion_Project/Optalysys/Batch_Analysis/',
                            # default='/media/erick/NuevoVol/LINUX_LAP/PhD/Optical_Correlation_Results/',
                            default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/',
                            filetypes='.mat', 
                            multiple='True')
    
    
    number_of_images, number_of_filters = gui.multenterbox(msg='How much images and filters?',
                                title='Number of images and filters',
                                fields=['Number of images:',
                                       'Number of filters:']) 
    
    number_of_images = int(number_of_images)
    number_of_filters = int(number_of_filters)
    
    #% Read MAT files
    CAMERA_PHOTO = scipy.io.loadmat(PATHS[0])
    _, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()
    
    INPUT_IMAGE_NUMBER = scipy.io.loadmat(PATHS[2])
    _, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values() 
    INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER[0, :]
    
    FILTER_IMAGE_NUMBER = scipy.io.loadmat(PATHS[1])
    _, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()
    FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER[0, :]
    
    COMB = np.transpose(np.vstack((INPUT_IMAGE_NUMBER, FILTER_IMAGE_NUMBER)))
    

else:
    
    PATHS = gui.fileopenbox(msg='Select File',
                            title='Files',
                            # default='/home/erick/Documents/PhD/Correaltion_Project/Optalysys/Batch_Analysis/',
                            # default='/media/erick/NuevoVol/LINUX_LAP/PhD/Optical_Correlation_Results/',
                            default='/media/erick/NuevoVol/LINUX_LAP/PhD/GT_200821/Cell_1/10um_150steps/1500 _Ecoli_HCB1_10x_50Hz_0.050ms_642nm_frame_stack_150steps_10um/',
                            filetypes='.npz', 
                            multiple='True')
    
    number_of_images, number_of_filters = gui.multenterbox(msg='How much images and filters?',
                                title='Number of images and filters',
                                fields=['Number of images:',
                                       'Number of filters:'])
    
    number_of_images = int(number_of_images)
    number_of_filters = int(number_of_filters)
    
    CAMERA_PHOTO = np.load(PATHS[0])
    CAMERA_PHOTO = CAMERA_PHOTO['a']
    
    a = np.arange(1, number_of_images+1)
    b = np.arange(1, number_of_filters+1)
    
    INPUT_IMAGE_NUMBER = np.repeat(a, repeats=number_of_filters)
    FILTER_IMAGE_NUMBER = np.tile(b, reps=number_of_images)
    
#% Order array according to image number combination with all filters
if mode:
    cam = np.empty_like(CAMERA_PHOTO)
    filter_num = np.empty_like(FILTER_IMAGE_NUMBER)   
    input_num = np.empty_like(INPUT_IMAGE_NUMBER)  
    
    for k in range(number_of_images):
        index_image = INPUT_IMAGE_NUMBER == k+1
        filter_num[k*number_of_filters:k*number_of_filters+number_of_filters] = FILTER_IMAGE_NUMBER[index_image]
        input_num[k*number_of_filters:k*number_of_filters+number_of_filters] = INPUT_IMAGE_NUMBER[index_image]
        cam[:, :, k*number_of_filters:k*number_of_filters+number_of_filters] = CAMERA_PHOTO[:, :, index_image]
    
    CAMERA_PHOTO = cam
    del cam
    
    comb = np.transpose(np.vstack((input_num, filter_num)))
    
#%% Export CAMERA_PHOTO as 2D txt
# ni, nj, nk = CAMERA_PHOTO.shape

# CC = np.empty((1, nj), dtype='uint8')
# for k in range(nk):
#     CC = np.vstack((CC, CAMERA_PHOTO[:, :, k]))
#     print(k)
# CC = CC[1:, :]    
# np.savetxt('F:\PhD\Archea_LW\Results_CES\correlation_2D.txt', CC, fmt='%i', delimiter=',')

#%%
# from skimage.feature import peak_local_max

# frame = 0
# # pk = peak_local_max(CAMERA_PHOTO[:,:,frame], min_distance=25, threshold_rel=0.6)
# pk = peak_local_max(CAMERA_PHOTO[:, :, frame], min_distance=25, threshold_abs=50)
# plt.imshow(CAMERA_PHOTO[:,:,frame])
# plt.scatter(pk[:, 1], pk[:, 0], c='r')

#%% Use SPT 2D trajectory to obtain 3D coordinates of correlation peaks



#%%
frame_number = np.arange(1, len(pks_vals)+1)
# pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number
max_val = np.amax(pks_vals, axis=1) 
max_val_locs = np.where((max_val != -1) & (max_val != -2))[0]  # index of rows of selected values different from -1 and -2

frame_val = frame_number[max_val_locs]
pks_vals = pks_vals[max_val_locs]
pksi = pksi[max_val_locs, :]
pksj = pksj[max_val_locs, :]

pks_max = pks_vals.max(axis=1,keepdims=1) == pks_vals  # mask of max values of pks_vals. Rows correspond to filter number

for i in range(len(pks_max)):
    if pks_max[i, :].all():
        pks_max[i, :] = False
            
image_match, filter_match = np.where(pks_max == True)
coord_i = pksi[image_match, filter_match]
coord_j = pksj[image_match, filter_match]
frame = frame_number[image_match]   



#%% CSAPS Smoothing
import functions as f

L = np.stack((coord_j, coord_i, filter_match), axis=1)
LL = pd.DataFrame(L, columns=['X', 'Y', 'Z'])

[x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 0.999999, True)
# [x_smooth, y_smooth, z_smooth] = f.csaps_smoothing(LL, 1, True)

#%% 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
#%matplotlib qt

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(coord_j, coord_i, filter_match, c=frame, label='Detected Positions')
ax.plot(x_smooth, y_smooth, z_smooth, c='red', label='Detected Positions')
# ax.plot(pos_i, pos_j, MAX_FILT, 'r-', label='Smoothed Curve')
pyplot.show()


#%% Plotly scatter plot
# import plotly.express as px
# import pandas as pd
# from plotly.offline import plot

# CURVE = pd.DataFrame(np.stack((x_smooth, y_smooth, z_smooth), axis=1), columns=['X', 'Y', 'Z'])

# fig = px.line_3d(CURVE, x='X', y='Y', z='Z')
# fig1 = px.scatter_3d(LL, x='X', y='Y', z='Z')

# fig.add_trace(fig1.data[0])

# fig.update_traces(marker=dict(size=1))
# plot(fig)

