import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

x = np.arange(-30, 30, .001)
x += 10**-50

plt.plot(x, (np.sin(x) / x) ** 2, 'r-')
plt.plot(x, (np.sin(x) / x), 'k-')
plt.title('Theoriekurve')
plt.grid(b = True, which = "major")
plt.xlabel(r'$\varphi \quad [10^{-3}\,\mathrm{rad}]$')
plt.ylabel('B, I auf 1 normiert')

plt.legend((r'Intensit\" at I($\varphi$)', r'Amplitude $B(\varphi)$'))

plt.savefig('theorie_1.png')