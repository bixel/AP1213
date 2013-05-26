import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab
from uncertainties import ufloat
from uncertainties import unumpy

# phi-Winkel
wellenlaenge, phiLinks, phiRechts = np.genfromtxt('messung_phi.txt', unpack = True)

phi = .5 * (phiRechts - phiLinks)

for i in range(0, np.size(phi)):
	print("phi = " + str(round(phi[i], 1)))

phiGesamt = np.sum(phi) / np.size(phi)
phiError = np.sqrt(1. / (np.size(phi) - 1.) * np.sum((phiGesamt - phi) ** 2))

print("phiGesamt = " + str(round(phiGesamt, 2)) + "\nphiError = " + str(round(phiError, 2)))

phiGesamtOptimal = 60.



# mu-Winkel
wellenlaenge, omegaLinks, omegaRechts = np.genfromtxt('messung_mu.txt', unpack = True)

mu = 180. - (omegaRechts - omegaLinks)
muOptimal = - 180. + (omegaRechts - omegaLinks)

for i in range(0, np.size(phi)):
	print("muOptimal = " + str(round(muOptimal[i], 2)))

n = np.sin((muOptimal + phiGesamt) * (np.pi / 180.) / 2.) / np.sin(phiGesamt * (np.pi / 180.) / 2.)

for i in range(0, np.size(n)):
	print(str(wellenlaenge[i]) + ": n = " + str(round(n[i], 3)))


nOptimal = np.sin((muOptimal + phiGesamtOptimal) * (np.pi / 180.) / 2.) / np.sin(phiGesamtOptimal * (np.pi / 180.) / 2.)

for i in range(0, np.size(n)):
	print(str(wellenlaenge[i]) + ": nOptimal = " + str(round(nOptimal[i], 3)))

dispersionsgleichung1 = lambda x, A0, A2, A4: A0 + A2 / x**2 + A4 / x**4
vorschlagswerte1 = np.array([1., 1., 1.])

dispersionsgleichung2 = lambda x, A2, A4: 1 - A2 * x**2 - A4 * x**4
vorschlagswerte2 = np.array([1., 1.])

nOptimal = nOptimal[::-1]
wellenlaenge *= 1e-9

koeffizienten1, varianzen1 = curve_fit(dispersionsgleichung1, wellenlaenge, nOptimal ** 2, p0 = vorschlagswerte1, maxfev = 1000)

koeffizienten2, varianzen2 = curve_fit(dispersionsgleichung2, wellenlaenge, nOptimal ** 2, p0 = vorschlagswerte2, maxfev = 1000)

print("A0 = " + str(koeffizienten1[0]) + " +- " + str(np.sqrt(varianzen1[0][0])))
print("A2 = " + str(koeffizienten1[1]) + " +- " + str(np.sqrt(varianzen1[1][1])))
print("A4 = " + str(koeffizienten1[2]) + " +- " + str(np.sqrt(varianzen1[2][2])))

s1quadrat = 1. / (np.size(wellenlaenge) - 2.) * np.sum((nOptimal ** 2 - koeffizienten1[0] - koeffizienten1[1] / wellenlaenge ** 2) ** 2)
print(s1quadrat)

s2quadrat = 1. / (np.size(wellenlaenge) - 2.) * np.sum((nOptimal ** 2 - koeffizienten2[0] + koeffizienten1[1] * wellenlaenge ** 2) ** 2)
print(s2quadrat)

# print("A2' = " + str(koeffizienten2[0]) + " +- " + str(np.sqrt(varianzen2[0][0])))
# print("A4' = " + str(koeffizienten2[1]) + " +- " + str(np.sqrt(varianzen2[1][1])))

print(koeffizienten2)
print(varianzen2)

wellenlaengeTheorie = np.arange(np.min(wellenlaenge), np.max(wellenlaenge), 1e-12)

plt.plot(wellenlaenge, nOptimal ** 2, "kx")
plt.plot(wellenlaengeTheorie, dispersionsgleichung1(wellenlaengeTheorie, koeffizienten1[0], koeffizienten1[1], koeffizienten1[2]), "r-")
plt.ylabel(r'$n^2$')
plt.xlabel(r'$\lambda \, [\mathrm{m}]$')
plt.legend(["Messwerte", "Dispersionsgleichung 1"])

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(10 * .7, 6 * .7)

plt.grid(which = 'both')
plt.savefig('../img/dispersion1.pdf')
plt.clf()


fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"

plt.plot(wellenlaenge, nOptimal ** 2, "kx")
plt.plot(wellenlaengeTheorie, dispersionsgleichung2(wellenlaengeTheorie, koeffizienten2[0], koeffizienten2[1]), "b-")

plt.ylabel(r'$n^2$')
plt.xlabel(r'$\lambda \, [\mathrm{m}]$')
plt.legend(["Messwerte", "Dispersionsgleichung 2"], "lower right")

fig.set_size_inches(10 * .7, 6 * .7)

plt.grid(which = 'both')
plt.savefig('../img/dispersion2.pdf')
plt.clf()

c = np.sqrt(dispersionsgleichung1(656e-9, koeffizienten1[0], koeffizienten1[1], koeffizienten1[2]))
d = np.sqrt(dispersionsgleichung1(589e-9, koeffizienten1[0], koeffizienten1[1], koeffizienten1[2]))
f = np.sqrt(dispersionsgleichung1(486e-9, koeffizienten1[0], koeffizienten1[1], koeffizienten1[2]))
print("nC = " + str(round(c, 4)))
print("nD = " + str(round(d, 4)))
print("nF = " + str(round(f, 4)))

abbe = (d - 1.) / (f - c)

print("abbe = " + str(round(abbe, 4)))

aufloesungC = - 3e-2 * (2 * koeffizienten1[1] / 656e-9**3 + 4 * koeffizienten1[2] / 656e-9**5)
aufloesungF = - 3e-2 * (2 * koeffizienten1[1] / 486e-9**3 + 4 * koeffizienten1[2] / 486e-9**5)

print(aufloesungC, aufloesungF)

print(koeffizienten1[2] / (koeffizienten1[0] - 1.), - koeffizienten1[1] / (koeffizienten1[0] - 1.))

