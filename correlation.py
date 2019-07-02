##
import math as m
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mpldatacursor
from progress.bar import Bar

##
I = mpimg.imread('MF1_30Hz_200us_awaysection.png')
IB = mpimg.imread('AVG_MF1_30Hz_200us_awaysection.png')

##
IFT = np.fft.fft2(I)
IFTS = np.fft.fftshift(IFT)

IBFT = np.fft.fft2(IB)
IBFTS = np.fft.fftshift(IBFT)

R = IFT*np.conj(IBFT)
r = np.real(np.fft.ifftshift(np.fft.ifft2(R)))

## Bokeh plot
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

## Pyplot plot
plt.figure(1)
plt.imshow(r, cmap='gray')
plt.colorbar()
mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),
                         formatter='x, y = {i}, {j}\nz = {z:.02g}'.format)
plt.show()

## Seaborn plot
# import seaborn as sns
#
# plt.figure(2)
# sns.heatmap(r, cmap='gray')
#

