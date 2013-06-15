import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

r, I, U = np.genfromtxt('abstand718.dat', unpack = True)

I = I / 1000
U = U / 1000

y = U * I
x = U / I

plt.plot(x, y, 'kx')

plt.savefig('plot_test.png', bbox_inches = 'tight')
plt.clf()

plt.plot(U, I, 'b-')
plt.savefig('plot_IU_test.png', bbox_inches = 'tight')
plt.clf()