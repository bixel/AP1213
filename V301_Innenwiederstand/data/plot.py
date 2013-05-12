import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

I, U = np.genfromtxt('messung_b.txt', unpack = True)

I /= 1000
Itheorie = np.arange(0, np.max(I) + .02, .001)
print(Itheorie)

fit_function = lambda x, m, b: m*x + b

koeffizienten, varianz = curve_fit(fit_function, I, U, maxfev = 1000)

plt.plot(I, U, "kx")
plt.plot(Itheorie, fit_function(Itheorie, koeffizienten[0], koeffizienten[1]), "r-")

print(koeffizienten[1])

I, U = np.genfromtxt('messung_c.txt', unpack = True)

I /= 1000 

fit_function = lambda x, m, b: m*x + b

koeffizienten, varianz = curve_fit(fit_function, I, U, maxfev = 1000)

plt.plot(I, U, "kx")
plt.plot(Itheorie, fit_function(Itheorie, koeffizienten[0], koeffizienten[1]), "r-")

print(koeffizienten[1])

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('graph.pdf')
plt.clf()