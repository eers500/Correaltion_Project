#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

IM = cv2.imread('R.png')
IM = IM[:, :, 0]

r = cv2.selectROI('IM', IM, False, False)
IMCROP = IM[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
# cv2.imshow('Image', imCrop)

IFILT = np.zeros_like(IM)
# A = np.pad(imCrop, 2, 'constant', constant_values=0)
nx = np.shape(IMCROP)[0]
ny = np.shape(IMCROP)[1]
IFILT[0:nx, 0:ny] = IMCROP

#%%
# IFT = np.fft.fft2(IN)
IFTS = np.fft.fft2(np.fft.fftshift(IM))

# IBFT = np.fft.fft2(IFILT)
IBFTS = np.fft.fft2(np.fft.fftshift(IFILT))

corr = IFTS*np.conj(IBFTS)
CORR = np.real(np.fft.ifft2(np.fft.fftshift(corr)))

#%%
# CORR = ndimage.correlate(IM, IMCROP, mode='reflect')

#%%
plt.imshow(np.abs(CORR), cmap='gray')