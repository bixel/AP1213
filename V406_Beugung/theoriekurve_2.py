import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

x = np.arange(-20, 20, .001)
x += 10**-50

y_ticks = np.arange(-.2, 1.3, .2)

plt.plot(x, (np.cos(x) * np.sin(x) / x) ** 2, 'r-')
plt.grid(b = True, which = "major")
plt.xlabel('x [mm]')
plt.yticks(y_ticks)
plt.ylabel(r'B, I normiert auf 1')

plt.legend((r'Intensit\" at I($\varphi$)', ''))

plt.savefig('theorie_2')
plt.show()