# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy import ndimage

IM = cv2.imread('R.png')
# IM = cv2.imread('NORM_MF1_30Hz_200us_awaysection.png')
IM = IM[:, :, 0]

IM_BINARY = np.zeros_like(IM)
IM_BINARY[IM > np.mean(IM)] = 255

# r = cv2.selectROI('IM', IM, False, False)
r = cv2.selectROI('IM_BINARY', IM_BINARY, False, False)

# IFILT = IM[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
IFILT = IM_BINARY[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# %%
# Correlation in Fourier space
FT = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
IFT = lambda X: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))

SI = np.shape(IM)
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

I_FT = FT(IM)
IFILT_FT = IFT(IPAD)

R = I_FT * np.conj(IFILT_FT)
r = np.real(IFT(R))

# %%
CORR = ndimage.correlate(IM, IFILT, mode='mirror')

# %%
# Pyplot plot
plt.figure(1)
plt.subplot(2, 2, 1); plt.imshow(IM_BINARY, cmap='gray'); plt.title('Hologram')
plt.subplot(2, 2, 2); plt.imshow(IFILT, cmap='gray'); plt.title('Mask')
plt.subplot(2, 2, 3); plt.imshow(CORR, cmap='gray'); plt.title('CORR')
plt.subplot(2, 2, 4); plt.imshow(r, cmap='gray'); plt.title('r')
f.dataCursor2D()
plt.show()
