import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

e, g_, B = np.genfromtxt("abbe.txt", unpack = True)
b_ = e - g_
G = 27.5
V = B / G

index = 0
while index < V.size:
	print(g_[index], b_[index], B[index], V[index])
	index += 1

funk = lambda x, f, h: f * x + h

vor_werte = np.array([100, 500])
p, cov = curve_fit(funk, (1 + 1/V), g_, p0 = vor_werte, maxfev = 1000)
print("f1 = " + str(p[0]) + "mm")
print("f1_fehler = " + str(np.sqrt(cov[0][0])) + "mm")
print("h = " + str(p[1]) + "mm")
print("h_fehler = " + str(np.sqrt(cov[1][1])) + "mm")


plt.plot(1 + 1/V, funk(1 + 1/V, p[0], p[1]), "r-")
plt.plot(1 + 1/V, g_, "kx")
plt.xlabel(r"$1 + 1/V$")
plt.ylabel(r"$g'\,[\mathrm{mm}]$")
plt.legend(('lineare Regression', 'Messwerte'), 'upper left')

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
#fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/graph_abbe_g.pdf', bbox_inches='tight')
plt.clf()

vor_werte = np.array([100, 500])
p, cov = curve_fit(funk, 1 + V, b_, p0 = vor_werte, maxfev = 1000)
print("f2 = " + str(p[0]) + "mm")
print("f2_fehler = " + str(np.sqrt(cov[0][0])) + "mm")
print("h_ = " + str(p[1]) + "mm")
print("h__fehler = " + str(np.sqrt(cov[1][1])) + "mm")

plt.plot(1 + V, funk(1 + V, p[0], p[1]), "r-")
plt.plot(1 + V, b_, "kx")
plt.xlabel(r"$1 + V$")
plt.ylabel(r"$b'\,[\mathrm{mm}]$")
plt.legend(('lineare Regression', 'Messwerte'), 'upper left')

plt.grid(which = 'both')
plt.savefig('../img/graph_abbe_b.pdf', bbox_inches='tight')
plt.clf()