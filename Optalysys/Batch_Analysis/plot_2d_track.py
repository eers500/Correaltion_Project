# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 17:17:33 2021

@author: eers500
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import easygui as gui

path = gui.fileopenbox()
#%%
pnumber = 3
dataf = pd.read_csv(path)
dataf = dataf[dataf['TRACK_ID'] == pnumber]
track = dataf[['POSITION_Y', 'POSITION_X', 'POSITION_T']].values

plt.figure(5)
plt.plot(track[:, 1], -track[:, 0])
plt.axis('tight')
plt.title('SPT 2D Track')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

