import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt('Rhodium_12s.txt')

plt.plot(x, y, 'kx')

plt.savefig('../img/graph_rhodium.png')