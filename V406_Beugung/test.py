import numpy as np
import scipy as sp
import pylab as pl
from scipy.optimize.minpack import curve_fit

x = np.array([  50.,  110.,  170.,  230.,  290.,  350.,  410.,  470.,  
530.,  590.])
y = np.array([ 3173.,  2391.,  1726.,  1388.,  1057.,   786.,   598.,   
443.,   339.,   263.])

smoothx = np.linspace(x[0], x[-1], 20)
guess_a, guess_b, guess_c = 4000, -0.005, 100
guess = [guess_a, guess_b, guess_c]

f_theory1 = lambda t, a, b, c: a * np.exp(b * t) + c

p, cov = curve_fit(f_theory1, x, y, p0=np.array(guess))

pl.clf()
f_fit1 = lambda t: p[0] * np.exp(p[1] * t) + p[2]
pl.plot(x, y, 'b.', smoothx, f_theory1(smoothx, guess_a, guess_b, guess_c))
pl.plot(x, y, 'b.', smoothx, f_fit1(smoothx), 'r-')
pl.show()