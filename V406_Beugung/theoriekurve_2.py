import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

x = np.arange(-3 * np.pi, 3 * np.pi, .001)
x += 10**-50

x_ticks = np.arange(-3 * np.pi, 4 * np.pi, np.pi/2)
x_labels = ('', '', '', '', '', '', '0', '', '', '', '', '', '')

#plt.plot(x, (np.sin(x) / x) ** 2, 'k-')
plt.plot(x, (np.cos(x) * np.sin(x) / x) ** 2, 'r-')
plt.title(r'$\displaystyle \frac{-3 \lambda}{b}$')
plt.grid(b = True, which = "major")
plt.xlabel('x [mm]')
plt.xticks(x_ticks, x_labels)
plt.ylabel('B, I [normiert auf 1]')

plt.savefig('theorie_2')
plt.show()