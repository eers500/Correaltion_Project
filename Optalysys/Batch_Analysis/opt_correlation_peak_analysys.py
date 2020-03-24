#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:13:47 2020

@author: erick
"""

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

IMAGES = scipy.io.loadmat('inputImages8Bit.mat')
_, _, _, IMAGES = IMAGES.values()
IMAGES = IMAGES[:, :, 0:21]
IMAGES = np.repeat(IMAGES, 21, axis=-1)

# IMAGES_F = scipy.io.loadmat('batchInput.mat')    # The bad ones
# _, _, _, IMAGES_F = IMAGES_F.values()

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
i = 21 # Image number
j = 21 # Filter number          
IMAGE_NUM = 21*(j-1)+i-1     # n*22, n=0,1,2... for image/filter pair
A = CAMERA_PHOTO[:, :, IMAGE_NUM] / np.max(CAMERA_PHOTO[:, :, IMAGE_NUM]) 
PKS = peak_local_max(A, num_peaks=10, min_distance=10)



k=0  # Peak number
DATA = A[PKS[k][0]-20:PKS[k][0]+20, PKS[k][1]-20:PKS[k][1]+20]
pks = peak_local_max(DATA, num_peaks=1)

def gauss(x, x0, y, y0, sigma, MAX):
    # return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    return MAX * np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))


I, J = np.meshgrid(np.arange(40), np.arange(40))
sig = np.linspace(0.1, 5, 200)
chisq = np.empty_like(sig)

for ii in range(np.shape(sig)[0]):
        chisq[ii] = np.sum((DATA - gauss(I, pks[0][0], J, pks[0][1], sig[ii], np.max(DATA)))**2)/np.var(DATA)

        
#%
LOC_MIN = np.where(chisq == np.min(chisq))
SIGMA_OPT = sig[LOC_MIN[0][0]]
ZZ = gauss(I, 20, J, 20, SIGMA_OPT, np.max(DATA))


plt.suptitle('Image number '+np.str(i)+' with Filter number '+np.str(j))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(IMAGES[:, :, IMAGE_NUM], cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Correlation with peaks marked')
plt.imshow(A)

for ii, txt in enumerate(np.arange(np.shape(PKS)[0])):
    plt.annotate(txt, (PKS[ii, 1], PKS[ii, 0]))
plt.scatter(PKS[:, 1], PKS[:, 0], marker='o', color='r', facecolors='none')

plt.subplot(2,2,3)
plt.title('Fitted Gaussian for peak '+np.str(k)+' and ' +r'$\sigma=$'+np.str(SIGMA_OPT.astype('float16')))
plt.imshow(ZZ)    

plt.subplot(2,2,4)
plt.title('Raw data for peak '+np.str(k))
plt.imshow(DATA)    
        
plt.show()