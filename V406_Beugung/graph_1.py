import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rc

x, y = np.genfromtxt('daten_mr_1.txt', unpack=True)
# Datensatz in x und y einlesen

x_theorie = np.arange(-25, 25.01, .001)
#x_theorie += 10**-9

x -= 25 + 10**-12
# x um 25 nach links verschieben (Maximum zentrieren)

y -= .13
# Dunkelstrom rausrechnen

y /= max(y)
# y normieren

x_ticks = np.arange(-28, 28.001, 4)
y_ticks = np.arange(0, 1.2, .1)
# Schritte fuer Skalierung festlegen

b_vorsch = .08
l = .000633

vor_werte = [b_vorsch]

theorie_funktion = lambda t, b: (np.sin(np.pi * b * (t / 1000) / l) / ((np.pi * b * (t / 1000)) / l)) ** 2

p, cov = curve_fit(theorie_funktion, x, y, p0 = np.array(vor_werte))

print(p)

plt.plot(x, y, 'kx')
plt.plot(x_theorie, theorie_funktion(x_theorie, p[0]), 'r-')
plt.title('Messreihe 1')
plt.xlabel(r"$\varphi \quad [10^{-3} \mathrm{rad}]$")
plt.ylabel(r"$I \quad [72\,\mathrm{nA}]$")
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(b = True, which = "major")

plt.legend(('Messpunkte', 'nicht-lineare Regression'), bbox_to_anchor=(1.1, 1))

plt.savefig('graph_1')