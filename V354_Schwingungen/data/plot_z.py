import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

# z_von_f = lambda f, R, L, C: np.sqrt(R**2 + (f*L - 1/(f*C))**2)
# x = np.arange(.1, 1000, .01)
# plt.plot(x, z_von_f(x, 50, 2, .01), "k-")
# plt.xscale("log")
# plt.ylabel(r"$|z|\,[\mathrm{\Omega}]$")
# plt.xlabel(r"$f\,[\mathrm{Hz}]$")

# fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
# fig.set_size_inches(9, 6)

# plt.grid(which = 'both')
# plt.savefig('../img/z_f.pdf', bbox_inches='tight')
# plt.clf()

theta = np.arange(0, 2 * np.pi + .1, .01)
angels = np.arange(0, 360, 45)
for i in range(5):
	plt.polar(theta, (i+1) + 0 * theta, "k-")
plt.thetagrids(angels)
rmax = np.arange(.001, i+1, .01)
for ang in angels:
	plt.polar(ang * (np.pi / 180) + 0 * rmax, rmax, "k-")

meintheta = np.arange(-np.pi/4, np.pi/4, .01)
plt.polar(meintheta, 3.5 / np.cos(meintheta), "r-")

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/z_phi.pdf', bbox_inches='tight')
plt.clf()