# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:26:49 2021

@author: eers500
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import easygui as gui
from tqdm import tqdm

path = gui.diropenbox()
write_path = gui.diropenbox()
file_list = os.listdir(path)

def create_filter(img, shape):
    # Padding
    SI = shape
    S = img.shape
    PAD = np.copy(img)
    PAD= np.pad(PAD, (int(np.floor(SI[0]/2)-np.floor(S[0]/2)), int(np.floor(SI[1]/2)-np.floor(S[1]/2))))
    
    # Phase
    FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
    IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))
    
    phase_sel = np.exp(1j**np.pi*PAD/255)
    filts = FT(phase_sel)
    f = -255*np.angle(filts)
    ff = np.zeros_like(f)
    ff[f >= 0] = 255
    
    # f = f-f.min()
    # ff = 255 * f/f.max()
    filt = np.uint8(ff)
    
    return filt

shape = [1000, 1000]
filters = []
for file in tqdm(file_list):
    img = plt.imread(path+'\\'+file)
    # filters.append(create_filter(img[:, :, 0], shape))
    filters.append(create_filter(img, shape))
    
for i in tqdm(np.arange(len(filters))):
    plt.imsave(write_path+'\\'+str(i)+'.png', filters[i], cmap='gray')
    
    
#%% Test correlation
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))
img = plt.imread(path+'\\'+file_list[0])
img = img[:, :, 0]
IM = FT(img)

PAD = img
phase_sel = np.exp(1j**np.pi*PAD/255)
filts = FT(phase_sel)
f = -255*np.angle(filts)
ff = np.zeros_like(f)
ff[f >= 0] = 255

# f = f-f.min()
# ff = 255 * f/f.max()
filt = np.uint8(ff)

C = np.real(IM*np.conj(filt))
plt.figure(1)
plt.imshow(C, cmap='jet')
plt.show()

#%
from scipy import ndimage
from scipy import signal

CC = signal.correlate2d(img, img)
plt.figure(2)
plt.imshow(CC, cmap='jet')


#%
plt.figure(3)
CCC = ndimage.filters.correlate(img, img)
plt.imshow(CCC, cmap='jet')

#%%
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

img = np.random.randint(0, 2, (256, 256))
IM = FT(img)

PAD = img
phase_sel = np.exp(1j**np.pi*PAD/255)
filts = FT(phase_sel)
f = -255*np.angle(filts)
ff = np.zeros_like(f)
ff[f >= 0] = 255
filt = np.uint8(ff)

C = np.real(IM*np.conj(filt))
plt.figure(1)
plt.imshow(C, cmap='jet')
plt.show()

from scipy import ndimage
CCC = ndimage.filters.correlate(img, img)
plt.figure(2)
plt.imshow(CCC, cmap='jet')
