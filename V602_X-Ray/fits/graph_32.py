import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt('../data/32_fein.txt', unpack = True)

x = 3084.183725 / np.sin( 2 * x * (np.pi / 360))

#np.sort(x)

index = np.where(y == max(y))

width = 4

m = (y[index[0] + width] - y[index[0]]) / (width / np.sin( 2 * .4 * (np.pi / 360)))
m = m[0]

print(y[index[0] + width])
print(y[index[0]])

b = y[index[0]] - (width * m);
b = b[0]

print(m)
print(b)

x_g = np.arange(10000, 11501, 100)

plt.plot(x, y, 'kx', x_g, m * x_g + y[index[0]], 'r-')
plt.xlabel('Energie')
plt.ylabel('Intensitaet')

plt.savefig('graph_32_Kante.png')
plt.clf() #clearfigure - damit wird der Graph zurueckgesetzt fuers naechste speichern