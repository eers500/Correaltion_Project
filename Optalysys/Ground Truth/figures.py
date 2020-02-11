import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')

P1 = np.loadtxt('particle1.txt')
P2 = np.loadtxt('particle2.txt')
P3 = np.loadtxt('particle3.txt')

# %%
fig, ax = plt.subplots(3, 2, sharex=True)

ax[0, 0].plot(P1[:, 0], P1[:, 0], 'k-', linewidth=2)
ax[0, 0].plot(P1[:, 0], P1[:, 3], 'bs', markersize=13, fillstyle='none', markeredgewidth=2)
ax[0, 0].tick_params(axis='both', direction='in', which='both', top=True, right=True)

ax[1, 0].plot(P2[:, 0], P2[:, 0], 'k-', linewidth=2)
ax[1, 0].plot(P2[:, 0], P2[:, 3], 'ro', markersize=13, fillstyle='none', markeredgewidth=2)
ax[1, 0].tick_params(axis='both', direction='in', which='both', top=True, right=True)

ax[2, 0].plot(P3[:, 0], P3[:, 0], 'k-', linewidth=2)
ax[2, 0].plot(P3[:, 0], P3[:, 3], 'g^', markersize=13, fillstyle='none', markeredgewidth=2)
ax[2, 0].tick_params(axis='both', direction='in', which='both', top=True, right=True)

ax[2, 0].set_xlabel('Z [$\mu$m]', fontsize=20)
ax[1, 0].set_ylabel('Z\' [$\mu$m]', fontsize=20)

plt.show()

#%%
from matplotlib.pylab import *
plt.figure()
plt.plot(P1[:, 0], P1[:, 0], 'k-', linewidth=5)
plt.plot(P1[:, 0], P1[:, 3], 'bs', markersize=40, fillstyle='none', markeredgewidth=5)
plt.tick_params(axis='both', direction='in', which='both', top=True, right=True, labelsize=30, )
rc('axes', linewidth=2)
plt.xlabel('Z [$\mu$m]', fontsize=40)
plt.ylabel('Z\' [$\mu$m]', fontsize=40)


plt.figure()
plt.plot(P2[:, 0], P2[:, 0], 'k-', linewidth=5)
plt.plot(P2[:, 0], P2[:, 3], 'ro', markersize=40, fillstyle='none', markeredgewidth=5)
plt.tick_params(axis='both', direction='in', which='both', top=True, right=True, labelsize=30)
rc('axes', linewidth=2)
plt.xlabel('Z [$\mu$m]', fontsize=40)
plt.ylabel('Z\' [$\mu$m]', fontsize=40)

plt.figure()
plt.plot(P3[:, 0], P3[:, 0], 'k-', linewidth=5)
plt.plot(P3[:, 0], P3[:, 3], 'g^', markersize=40, fillstyle='none', markeredgewidth=5)
plt.axis('square')
plt.tick_params(axis='both', direction='in', which='both', top=True, right=True, labelsize=30)
rc('axes', linewidth=2)
plt.xlabel('Z [$\mu$m]', fontsize=40)
plt.ylabel('Z\' [$\mu$m]', fontsize=40)


plt.show()