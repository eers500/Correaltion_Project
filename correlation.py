#%%
# import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mpldatacursor
from scipy import ndimage
from progress.bar import Bar

#%%
INPUT = mpimg.imread('MF1_30Hz_200us_awaysection.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
IFILT = mpimg.imread('MF1_1.png')
# IFILT = IFILT[:, :, 0]
I = INPUT/IB

# CORR = ndimage.correlate(IN, I_FILT, mode='wrap')
#%%
FT  =  lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT =  lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

SI = np.shape(I)
S = np.shape(IFILT)
NY1 = int(np.floor((SI[0]-S[0])/2))
NX1 = int(np.floor((SI[0]-S[1])/2))
IPAD = np.pad(IFILT, ((NX1, NX1), (NY1, NY1)), 'constant', constant_values=0)

I_FT = FT(I)
IFILT_FT = IFT(IPAD)

R = I_FT*np.conj(IFILT_FT)
r = np.real(IFT(R))

#%%
# Pyplot plot
plt.figure(1)
plt.imshow(r, cmap='gray')
plt.colorbar()
mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                         formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
plt.show()

#%%
# # 3D surace Plot
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
# # ax.set_title('Cells Positions in 3D', fontsize='20')
# # ax.set_xlabel('x (pixels)', fontsize='18')
# # ax.set_ylabel('y (pixels)', fontsize='18')
# # ax.set_zlabel('z (slices)', fontsize='18')
#
# MAX = np.mean(CORR)*np.ones_like(X)
# MAX[xi[0], yi[0]] = np.max(CORR)
# ax.plot_surface(X, Y, MAX)
# pyplot.show()
