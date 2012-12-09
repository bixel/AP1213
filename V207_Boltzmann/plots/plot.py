import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x = np.arange(1e-12, 10e-6 + 1e-12, .01e-6)
x2 = np.arange(-10 + .01, 10 + .01, .1)

c = 299792458.
k = 1.3806503e-23
T = 2930.15
h = 6.62606876e-34
ch = 1.98644544e-25

function = lambda l: 1 / 1e9 * 1 / 2 * (c * ch / (l ** 5)) / (np.e ** (ch / (k * l * T)) - 1)
#function2 = lambda l: 1/ l ** 5 / (np.e ** (1 / l) - 1 )

plt.plot(x, function(x), 'k-')
plt.savefig("plot.pdf")
plt.clf()