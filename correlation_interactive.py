# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy import ndimage

IM = cv2.imread('R_Binary.png')
IM1 = cv2.imread('R.png')
# IM = cv2.imread('NORM_MF1_30Hz_200us_awaysection.png')
IM = IM[:, :, 0]
IM1 = IM1[:, :, 0]

IM_BINARY = np.zeros_like(IM1)
IM_BINARY[IM > np.mean(IM1)] = 255

# fig, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')
# ax1.imshow(IM, cmap='gray')
# ax2.imshow(IM_BINARY, cmap='gray')
# f.dataCursor2D()
#%%
# S = cv2.selectROI('IM', IM, False, False)
S = cv2.selectROI('IM_BINARY', IM_BINARY, False, False)

# IFILT = IM[int(S[1]):int(S[1] + S[3]), int(S[0]):int(S[0] + S[2])]
IFILT = IM_BINARY[int(S[1]):int(S[1] + S[3]), int(S[0]):int(S[0] + S[2])]


# %%
# Correlation in Fourier space
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

SI = np.shape(IM_BINARY)
S = np.shape(IFILT)
DSY = SI[0] - S[0]
DSX = SI[1] - S[1]

if DSY % 2 == 0 and DSX % 2 == 0:
    NY = int(DSY / 2)
    NX = int(DSX / 2)
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX)), 'constant', constant_values=0)
elif DSY % 2 == 1 and DSX % 2 == 1:
    NY = int(np.floor(DSY / 2))
    NX = int(np.floor(DSX / 2))
    IPAD = np.pad(IFILT, ((NY, NY + 1), (NX, NX + 1)), 'constant', constant_values=0)
elif DSY % 2 == 0 and DSX % 2 == 1:
    NY = int(DSY / 2)
    NX = int(np.floor(DSX / 2))
    IPAD = np.pad(IFILT, ((NY, NY), (NX, NX + 1)), 'constant', constant_values=0)
elif DSY % 2 == 1 and DSX % 2 == 0:
    NY = int(np.floor(DSY / 2))
    NX = int(DSX / 2)
    IPAD = np.pad(IFILT, ((NY, NY + 1), (NX, NX)), 'constant', constant_values=0)

# I_FT = FT(IM)
I_FT = FT(IM_BINARY)
IFILT_FT = IFT(IPAD)

R = I_FT * np.conj(IFILT_FT)
r = np.abs(IFT(R))**2

# %%
CORR = ndimage.correlate(IM_BINARY, IFILT, mode='wrap')
# CORR = ndimage.correlate(IM, IFILT, mode='wrap')

#%%
# Pyplot plot
ax1 = plt.subplot(2, 2, 1);  plt.imshow(IM_BINARY, cmap='gray'); plt.title('Hologram')
ax2 = plt.subplot(2, 2, 2); plt.imshow(IFILT, cmap='gray'); plt.title('Mask')
ax3 = plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1);  plt.imshow(CORR, cmap='gray'); plt.title('CORR')
ax4 = plt.subplot(2, 2, 4, sharex=ax1, sharey=ax1); plt.imshow(r, cmap='gray'); plt.title('r')
f.dataCursor2D()
plt.show()

