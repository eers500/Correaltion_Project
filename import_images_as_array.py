#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:15:46 2019

@author: erick
"""

import glob
#import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#%%
#images = [cv2.imread(file) for file in glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png")]

#%%
images = [mpimg.imread(file) for file in np.sort(glob.glob("/home/erick/Documents/PhD/23_10_19/LUT_MANUAL/*.png"))]
images = np.swapaxes(np.swapaxes(images, 0, 1), 1, 2)

#%%
plt.imshow(images[:, :, 0], cmap='gray')
plt.show()