import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

x = np.arange(-30, 30, .001)
x += 10**-50

x_ticks = np.arange(-30, 31, 10)
y_ticks = np.arange(-.2, 1.3, .2)

s = .16

b = .04

l = .000633

plt.plot(x, np.cos(np.pi * s * (x / 1000) / l) ** 2 * (np.sin(np.pi * b * (x / 1000) / l) / ((np.pi * b * (x / 1000)) / l)) ** 2, 'r-')
plt.grid(b = True, which = "major")
plt.xticks(x_ticks)
plt.xlabel(r'$\varphi \quad [10^{-3}\,\mathrm{rad}]$')
plt.yticks(y_ticks)
plt.ylabel(r'B, I normiert auf 1')

plt.legend((r'Intensit\" at I($\varphi$)', ''))

plt.savefig('theorie_2.png')