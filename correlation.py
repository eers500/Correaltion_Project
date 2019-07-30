#%%
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mpldatacursor
from scipy import ndimage
from progress.bar import Bar

#%%
I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')
I_FILT = mpimg.imread('MF1_1.png')
IN = I/IB

CORR = ndimage.correlate(IN, I_FILT, mode='wrap')
#%%
IFT = np.fft.fft2(IN)
IFTS = np.fft.fftshift(IFT)

NPAD = np.uint8((len(IN)-len(I_FILT))/2)
IFILT = np.pad(I_FILT, ((NPAD, NPAD), (NPAD, NPAD)), 'constant')
IBFT = np.fft.fft2(IFILT)
IBFTS = np.fft.fftshift(IBFT)

R = IFT*np.conj(IBFT)
r = np.real(np.fft.ifftshift(np.fft.ifft2(R)))

#%%
# Bokeh plot
# from bokeh.plotting import figure, show, output_file
#
# p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
# # p.x_range.range_padding = p.y_range.range_padding = 0
#
# # must give a vector of image data for image parameter
# im = r
# p.image(image=[im[::-1]], x=0, y=0, dw=512, dh=512, palette="Spectral11")
#
# output_file("image.html", title="image.py example")
#
# show(p)  # open a browser

#%%
# Pyplot plot
plt.figure(1)
plt.imshow(CORR, cmap='gray')
plt.colorbar()
mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                         formatter='x, y = {i}, {j}\nz = {z:.06g}'.format)
plt.show()

#%%
# Seaborn plot
# import seaborn as sns
#
# plt.figure(2)
# sns.heatmap(r, cmap='gray')

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
