#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:13:47 2020

@author: erick
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.feature import peak_local_max

#%%
CAMERA_PHOTO = scipy.io.loadmat('camera_photo.mat')
_, _, _, CAMERA_PHOTO = CAMERA_PHOTO.values()

INPUT_IMAGE_NUMBER = scipy.io.loadmat('input_image_number.mat')
_, _, _, INPUT_IMAGE_NUMBER = INPUT_IMAGE_NUMBER.values()

FILTER_IMAGE_NUMBER = scipy.io.loadmat('filter_image_number.mat')
_, _, _, FILTER_IMAGE_NUMBER = FILTER_IMAGE_NUMBER.values()

A, _ = np.where(INPUT_IMAGE_NUMBER > 21)
CAMERA_PHOTO = np.delete(CAMERA_PHOTO, A, axis=-1)
INPUT_IMAGE_NUMBER = np.delete(INPUT_IMAGE_NUMBER, A)
FILTER_IMAGE_NUMBER = np.delete(FILTER_IMAGE_NUMBER , A)

Z = np.argsort(INPUT_IMAGE_NUMBER[0:21])
ZZ = CAMERA_PHOTO[:, :, Z]
ZZZ = INPUT_IMAGE_NUMBER[0:21]
ZZZZ = ZZZ[Z]

CAMERA_PHOTO[:, :, 0:21] = ZZ
INPUT_IMAGE_NUMBER[0:21] = ZZZZ
del Z, ZZ, ZZZ, ZZZZ

CORR_CPU = np.load('CORR_CPU.npy')
CAMERA_PHOTO = CAMERA_PHOTO[380:856, 604:1100, :].astype('float32')

#%%
A = CAMERA_PHOTO[:, :, 22*0]
PKS = peak_local_max(A, num_peaks=6, min_distance=10)

plt.imshow(A)
plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', color='r', facecolors='none')
plt.show()

#%%
import scipy.optimize

k=5
DATA = A[PKS[k][0]-20:PKS[k][0]+20, PKS[k][1]-20:PKS[k][1]+20]
pks = peak_local_max(DATA, num_peaks=1)

def gauss(x, x0, y, y0, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


I, J = np.meshgrid(np.arange(40), np.arange(40))
sig = np.linspace(0.1, 5, 200)
chisq = np.empty_like(sig)

for i in range(np.shape(sig)[0]):
        chisq[i] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[i]))**2)/np.var(DATA)
        
#plt.plot(sig, -chisq)
#plt.grid()
        
#%
LOC_MIN = np.where(-chisq == np.min(-chisq))
SIG_OPT = sig[LOC_MIN[0][0]]
ZZ = gauss(I, 20, J, 20, SIG_OPT)

plt.subplot(1,2,1)
plt.imshow(ZZ)    

plt.subplot(1,2,2)
plt.imshow(DATA)    
        