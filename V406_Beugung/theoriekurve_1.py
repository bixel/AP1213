import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-30, 30, .001)
x += 10**-50

plt.plot(x, (np.sin(x) / x) ** 2, 'r-')
plt.plot(x, (np.sin(x) / x), 'k-')
plt.title('Theoriekurve')
plt.grid(b = True, which = "major")
plt.xlabel('x [mm]')
plt.ylabel('B, I [normiert auf 1]')

plt.savefig('theorie_1')
plt.show()