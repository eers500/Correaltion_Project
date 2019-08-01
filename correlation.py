#%%
# import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy import ndimage
from progress.bar import Bar

#%%
INPUT = mpimg.imread('R.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
IFILT = mpimg.imread('SampleRing.png')
# IFILT = IFILT[:, :, 0]
I = INPUT

CORR = ndimage.correlate(I, IFILT, mode='wrap')
#%%
# Correlation in Fourier space
FT  =  lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT =  lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

SI = np.shape(I)
S = np.shape(IFILT)
DSY = SI[0]-S[0]
DSX = SI[1]-S[1]

if DSY % 2 == 0 and DSX % 2 == 0:
    NY = DSY/2
    NX = DSX/2
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX)), 'constant', constant_values=0)
elif DSY % 2 == 0 and DSX % 2 == 1:
    NY = DSY/2
    NX = np.floor(DSX/2)
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX+1)), 'constant', constant_values=0)
elif DSY % 2 == 1 and DSX % 2 == 0:
    NY = int(np.floor(DSY/2))
    NX = int(DSX/2)
    IPAD = np.pad(IFILT, ((NY, NY+1), (NX, NX)), 'constant', constant_values=0)

I_FT = FT(I)
IFILT_FT = IFT(IPAD)

R = I_FT*np.conj(IFILT_FT)
r = np.real(IFT(R))

#%%
# Pyplot plot
plt.figure(1)
plt.subplot(2, 2, 1);  plt.imshow(I, cmap='gray'); plt.title('Hologram')
plt.subplot(2, 2, 2); plt.imshow(IFILT, cmap='gray'); plt.title('Mask')
plt.subplot(2, 2, 3);  plt.imshow(CORR, cmap='gray'); plt.title('CORR')
plt.subplot(2, 2, 4); plt.imshow(r, cmap='gray'); plt.title('r')
f.dataCursor()
plt.show()

#%%
# 3D surace Plot
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot
#
# xi, yi = np.where(CORR == np.max(CORR))
#
# fig = pyplot.figure()
# ax = Axes3D(fig)
#
# X, Y = np.meshgrid(np.arange(1, 513, 1), np.arange(1, 513, 1))
# ax.plot_surface(X, Y, CORR)
# ax.tick_params(axis='both', labelsize=10)
# ax.set_title('Cells Positions in 3D', fontsize='20')
# ax.set_xlabel('x (pixels)', fontsize='18')
# ax.set_ylabel('y (pixels)', fontsize='18')
# ax.set_zlabel('z (slices)', fontsize='18')
#
# # MAX = np.mean(CORR)*np.ones_like(X)
# # MAX[xi[0], yi[0]] = np.max(CORR)
# # ax.plot_surface(X, Y, MAX)
# pyplot.show()
