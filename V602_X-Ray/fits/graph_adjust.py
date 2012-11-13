import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt('../data/adjust.txt', unpack = True)

x_half = (27.21, 28.9)
y_half = (113.5, 113.5)

plt.plot(x_half, y_half, 'kx')
plt.plot(x, y, 'r-')
plt.xlabel(r'$\theta \quad [\degree]$')
plt.ylabel('Intensitaet')

plt.grid(b = True, which = "major") #Grid aktivieren

#I-0 Linie
plt.annotate('', xy = (26, 36), xycoords = 'data', xytext = (30, 38), textcoords = 'data', arrowprops = dict(arrowstyle = "-"))

#Horizontaler Pfeil
plt.annotate('', xy=(27.21, 113.5),  xycoords='data', xytext=(27.75, 113.5), textcoords='data', arrowprops=dict(arrowstyle="->"))
plt.annotate('', xy=(28.9, 113.5), xycoords = 'data', xytext=(28.25, 113.5), textcoords='data', arrowprops=dict(arrowstyle="->"))
plt.annotate(r'$\theta_{1 / 2}$', xy = (27.9, 111.5), xycoords = 'data', arrowprops = None)

#Vertikale Linien
plt.vlines(27.21, 20, 189, color = 'k', linestyle = '--')
plt.vlines(28.9, 20, 113.5, color = 'k', linestyle = '--')
#plt.hlines(113.5, 27.21, 28.9, color = 'k', linestyle = '--')

#Veritkaler Pfeil
plt.annotate('', xy=(28.9, 189),  xycoords='data', xytext=(28.9, 145), textcoords='data', arrowprops=dict(arrowstyle="->"))
plt.annotate('', xy=(28.9, 113.5), xycoords = 'data', xytext=(28.9, 130), textcoords='data', arrowprops=dict(arrowstyle="->"))
plt.annotate(r'$I_{\mathrm{max}} / 2$', xy = (28.9, 135), xycoords = 'data', arrowprops = None)


fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 4.5)

plt.savefig('graph_adjust.png')
plt.clf() #clearfigure - damit wird der Graph zurueckgesetzt fuers naechste speichern
fig.clf()