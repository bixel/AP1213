import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

y, x = np.genfromtxt('tabelle501a.dat', unpack=True)
# Datensatz in x und y einlesen

b_vorsch = 1
a_vorsch = 14.075

x = 1/x

y /= 4

x[::-1]

x_theorie = np.arange(min(x), max(x), .00001)

vor_werte = [b_vorsch, a_vorsch]

theorie_funktion = lambda t, b, a: a*t + b

p, cov = curve_fit(theorie_funktion, x, y, p0 = np.array(vor_werte), maxfev = 3000)

print(p)

plt.plot(x, y, 'kx')
plt.plot(x_theorie, theorie_funktion(x_theorie, p[0], p[1]), 'r-')
plt.title('Messreihe 3')
plt.xlabel('x [mm]')
plt.ylabel('I [nA]')
plt.grid(b = True, which = "major")

plt.legend(('Messpunkge', 'nicht-lineare Regression'))

plt.savefig('graph_2')
plt.show()