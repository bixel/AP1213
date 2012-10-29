import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

x, y = np.genfromtxt('daten_mr_2.txt', unpack=True)
# Datensatz in x und y einlesen

x_theorie = np.arange(-25, 25.01, .001)
#x_theorie += 10**-9

x -= 25 + 10**-12
# x um 25 nach links verschieben (Maximum zentrieren)

y -= .13
# Dunkelstrom rausrechnen

y /= max(y)
# y normieren

x_ticks = np.arange(-25, 26, 5)
y_ticks = np.arange(0, 1.2, .1)
# Schritte fuer Skalierung festlegen

b_vorsch = .02
l = .000633

vor_werte = [b_vorsch]

theorie_funktion = lambda t, b: (np.sin(np.pi * b * (t / 1000) / l) / ((np.pi * b * (t / 1000)) / l)) ** 2

p, cov = sp.optimize.curve_fit(theorie_funktion, x, y, p0 = np.array(vor_werte))

print(p)

plt.plot(x, y, 'kx')
plt.plot(x_theorie, theorie_funktion(x_theorie, p[0]), 'r-')
plt.title('Messreihe 2')
plt.xlabel('x [mm]')
plt.ylabel('I [nA]')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(b = True, which = "major")

plt.legend(('Messpunkge', 'nicht-lineare Regression'))

plt.savefig('graph_2')
plt.show()