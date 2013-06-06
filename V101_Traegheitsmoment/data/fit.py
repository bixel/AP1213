import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

a, T, N = np.genfromtxt('messung2_T.txt', unpack = True)

T /= N

for i in range(0, np.size(T)):
	print(np.round(T[i], 2))

a *= .01
#D = 0.000184899164944
D = 0.0199411446038
m = .44593

#fitting_funktion = lambda a, B, C: (4 * np.pi**2) / C * (B + 2 * (.058 / 1000 + m * a**2))
#fitting_funktion = lambda a, B, C: B + C * 2 * (.058 / 1000 + m * a**2)
#fitting_funktion = lambda a, B: B + (4 * np.pi**2) / 0.0199411446038 * 2 * (.058 / 1000 + m * a**2)
fitting_funktion = lambda a, M, B: M * a ** 2 + B

vorschlagswerte = np.array([1, 1])
fitting_durchlaeufe = 1000
parameter, varianz = curve_fit(fitting_funktion, a, T ** 2, p0 = vorschlagswerte, maxfev = fitting_durchlaeufe)

print(parameter)
# print("B = " + str(parameter[0]) + "gm^2")
# print(np.sqrt(varianz[0][0]))
# print("D = " + str(parameter[1]) + "gm^2")
# print(np.sqrt(varianz[1][1]))
print(8 * np.pi ** 2 / parameter[0] * m / 2)


plt.plot(a ** 2, T ** 2, "kx")
plt.plot(a ** 2, fitting_funktion(a, parameter[0], parameter[1]), "r-")
plt.xlabel(r"$a^2 [\mathrm{m}^2]$")
plt.ylabel(r"$T^2 [\mathrm{s}^2]$")

plt.legend(np.array(["Messpunkte", "Ausgleichsgerade"]), "upper left")

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/graph2_T.pdf')
plt.clf()