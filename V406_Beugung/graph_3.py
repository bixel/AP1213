import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt('daten_mr_3.txt', unpack=True)
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

b_vorsch = .04
s_vorsch = .250
l = .000633

vor_werte = [b_vorsch, s_vorsch]

theorie_funktion = lambda t, b, s: np.cos(np.pi * s * (t / 1000) / l) ** 2 * (np.sin(np.pi * b * (t / 1000) / l) / ((np.pi * b * (t / 1000)) / l)) ** 2
theorie_funktion_2 = lambda t, b: (np.sin(np.pi * b * (t / 1000) / l) / ((np.pi * b * (t / 1000)) / l)) ** 2

p, cov = curve_fit(theorie_funktion, x, y, p0 = np.array(vor_werte), maxfev = 2000)
p2, cov2 = curve_fit(theorie_funktion, x, y, p0 = np.array(vor_werte), maxfev = 2000)

print(p)

plt.plot(x, y, 'kx')
line1 = plt.plot(x_theorie, theorie_funktion(x_theorie, p[0], p[1]))

plt.setp(line1, linewidth = .5, color = 'r', ls = '-', aa = True)

plt.plot(x_theorie, theorie_funktion_2(x_theorie, p2[0]), 'b--')
plt.title('Messreihe 3')
plt.xlabel(r"$\varphi \quad [10^{-3} \mathrm{rad}]$")
plt.ylabel(r"$I \quad [38\,\mathrm{nA}]$")
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(b = True, which = "major")

plt.legend(('Messpunkte', 'nicht-lineare Regression', 'Einzelspaltmuster'), bbox_to_anchor=(1.1, 1))

fig = plt.gcf()
fig.set_size_inches(9, 6)

plt.savefig('graph_3.png', pad_inches = 0)