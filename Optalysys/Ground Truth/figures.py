import numpy as np
import matplotlib.pyplot as plt

P1 = np.loadtxt('particle1.txt')
P2 = np.loadtxt('particle2.txt')
P3 = np.loadtxt('particle3.txt')

#%%
plt.plot(P1[:, 0], P1[:, 0])
plt.plot(P1[:, 0], P1[:, 3], 'rs')
plt.xlabel('Plate Position [$\mu$m]')
plt.ylabel('DZ [$\mu$m]')
plt.show()
