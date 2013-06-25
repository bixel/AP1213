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

#print("phiGesamt = " + str(round(phiGesamt, 2)) + "\nphiError = " + str(round(phiError, 2)))

phiGesamtOptimal = 60.



# mu-Winkel
wellenlaenge, omegaLinks, omegaRechts = np.genfromtxt('messung_mu.txt', unpack = True)

mu = 180. - (omegaRechts - omegaLinks)
muOptimal = - 180. + (omegaRechts - omegaLinks)

for i in range(0, np.size(phi)):
	print("muOptimal = " + str(round(muOptimal[i], 2)))

n = np.sin((muOptimal + phiGesamt) * (np.pi / 180.) / 2.) / np.sin(phiGesamt * (np.pi / 180.) / 2.)
deltaN = np.sqrt(( np.sin(muOptimal / 2. * (np.pi / 180.)) / (np.cos(phiGesamt * (np.pi / 180.)) - 1) )**2) * phiError * (np.pi / 180.)

for i in range(0, np.size(n)):
	print(str(wellenlaenge[i]) + ": n = " + str(round(n[i], 3)) + "+-" + str(round(deltaN[i], 5)))


nOptimal = np.sin((muOptimal + phiGesamt) * (np.pi / 180.) / 2.) / np.sin(phiGesamt * (np.pi / 180.) / 2.)

for i in range(0, np.size(n)):
	print(str(wellenlaenge[i]) + ": nOptimal = " + str(round(nOptimal[i], 3)))

dispersionsgleichung1 = lambda x, A0, A2: A0 - A2 * x
vorschlagswerte1 = np.array([1., 1.])

dispersionsgleichung2 = lambda x, A0, A2: A0 + A2 * x
vorschlagswerte2 = np.array([1., 1.])

#nOptimal = nOptimal[::-1]
wellenlaenge *= 1e-9

koeffizienten1, varianzen1 = curve_fit(dispersionsgleichung1, wellenlaenge ** 2, nOptimal ** 2, p0 = vorschlagswerte1, maxfev = 1000)

koeffizienten2, varianzen2 = curve_fit(dispersionsgleichung2, 1 / wellenlaenge ** 2, nOptimal ** 2, p0 = vorschlagswerte2, maxfev = 1000)

print("A0' = " + str(koeffizienten1[0]) + " +- " + str(np.sqrt(varianzen1[0][0])))
print("A2' = " + str(koeffizienten1[1]) + " +- " + str(np.sqrt(varianzen1[1][1])))

s1quadrat = 1. / (np.size(wellenlaenge) - 2.) * np.sum((nOptimal ** 2 - koeffizienten1[0] - koeffizienten2[1] / wellenlaenge ** 2) ** 2)
print("s'^2 = " + str(s1quadrat))

s2quadrat = 1. / (np.size(wellenlaenge) - 2.) * np.sum((nOptimal ** 2 - koeffizienten2[0] + koeffizienten1[1] * wellenlaenge ** 2) ** 2)
print("s^2 = " + str(s2quadrat))

print("A0 = " + str(koeffizienten2[0]) + " +- " + str(np.sqrt(varianzen2[0][0])))
print("A2 = " + str(koeffizienten2[1]) + " +- " + str(np.sqrt(varianzen2[1][1])))

#print(koeffizienten2)
#print(varianzen2)

wellenlaengeTheorie = np.arange(np.min(wellenlaenge), np.max(wellenlaenge), 1e-12)
dispersionsgleichungOptimal = lambda x: np.sqrt(koeffizienten2[0] + koeffizienten2[1] / x**2)

plt.plot(wellenlaenge ** 2, nOptimal ** 2, "kx")
plt.plot(wellenlaengeTheorie ** 2, dispersionsgleichung1(wellenlaengeTheorie ** 2, koeffizienten1[0], koeffizienten1[1]), "r-")
plt.ylabel(r'$n^2$')
plt.xlabel(r'$1 / \lambda^2 \, [1 / \mathrm{m}^2]$')
plt.legend(["Messwerte", "Dispersionsgleichung 2"])

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(10 * .7, 7 * .7)

plt.grid(which = 'both')
plt.savefig('../img/dispersion2falsch.pdf')
plt.clf()


fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"

plt.plot(1 / wellenlaenge ** 2, nOptimal ** 2, "kx")
plt.plot(1 / wellenlaengeTheorie ** 2, dispersionsgleichung2(1 / wellenlaengeTheorie ** 2, koeffizienten2[0], koeffizienten2[1]), "b-")

plt.ylabel(r'$n^2$')
plt.xlabel(r'$\lambda^2 \, [\mathrm{m}^2]$')
plt.legend(["Messwerte", "Dispersionsgleichung 1"], "lower right")

fig.set_size_inches(10 * .7, 6.5 * .7)

plt.grid(which = 'both')
plt.savefig('../img/dispersion1falsch.pdf')
plt.clf()

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"

#nOptimal = nOptimal[::-1]

plt.plot(wellenlaenge, nOptimal, "kx")
plt.plot(wellenlaengeTheorie, dispersionsgleichungOptimal(wellenlaengeTheorie), "b-")

plt.ylabel(r'$n$')
plt.xlabel(r'$\lambda \, [\mathrm{m}]$')
plt.legend(["Messwerte", "Dispersionsgleichung 1"], "lower right")

fig.set_size_inches(10 * .7, 6.5 * .7)

plt.grid(which = 'both')
plt.savefig('../img/dispersionNichtLinear.pdf')
plt.clf()

lambdaC = 656e-9
lambdaD = 589e-9
lambdaF = 486e-9

c = np.sqrt(dispersionsgleichung2(1/lambdaC ** 2, koeffizienten1[0], koeffizienten2[1]))
deltaC = .5 * np.sqrt(1/(koeffizienten2[0] + koeffizienten2[1] / lambdaC**2) * (varianzen2[0][0] + varianzen2[1][1] / lambdaC**4))
d = np.sqrt(dispersionsgleichung2(1/lambdaD ** 2, koeffizienten1[0], koeffizienten2[1]))
deltaD = .5 * np.sqrt(1/(koeffizienten2[0] + koeffizienten2[1] / lambdaD**2) * (varianzen2[0][0] + varianzen2[1][1] / lambdaD**4))
f = np.sqrt(dispersionsgleichung2(1/lambdaF ** 2, koeffizienten1[0], koeffizienten2[1]))
deltaF = .5 * np.sqrt(1/(koeffizienten2[0] + koeffizienten2[1] / lambdaF**2) * (varianzen2[0][0] + varianzen2[1][1] / lambdaF**4))
print("nC = " + str(round(c, 5)) + "+-" + str(round(deltaC,5)))
print("nD = " + str(round(d, 5)) + "+-" + str(round(deltaD,5)))
print("nF = " + str(round(f, 5)) + "+-" + str(round(deltaF,5)))

abbe = (d - 1.) / (f - c)

print("abbe = " + str(round(abbe, 4)) + "+-" + str(round(np.sqrt((1/(f-c) * deltaD) ** 2 + ((1-d)/((f-c)**2)*deltaF) ** 2 + ((d-1)/((f-c)**2)*deltaC) ** 2),4)))

b = 3e-2
A0 = koeffizienten1[0]
deltaA0 = np.sqrt(varianzen2[0][0])
A2 = koeffizienten2[1]
deltaA2 = np.sqrt(varianzen2[1][1])
aufloesungC = b * (A2 / lambdaC**3) * 1. / np.sqrt(A0 + A2 / lambdaC**2)
deltaAufloesungC = np.sqrt((b * A2 / (2. * lambdaC**3 * (A0 + A2 / lambdaC**2) ** 1.5))**2 * deltaA0**2 + (b / (lambdaC**3 *np.sqrt(A0 + A2 / lambdaC ** 2)) - b * A2 / (lambdaC ** 5 * np.sqrt(A0 + A2 / lambdaC ** 2) ** 3))**2 * deltaA2**2)
aufloesungF = b * (A2 / lambdaF**3) * 1. / np.sqrt(A0 + A2 / lambdaF**2)
deltaAufloesungF = np.sqrt((b * A2 / (2. * lambdaF**3 * (A0 + A2 / lambdaF**2) ** 1.5))**2 * deltaA0**2 + (b / (lambdaF**3 *np.sqrt(A0 + A2 / lambdaF ** 2)) - b * A2 / (lambdaF ** 5 * np.sqrt(A0 + A2 / lambdaF ** 2) ** 3))**2 * deltaA2**2)

print("AC = " + str(round(aufloesungC)) + "+-" + str(round(deltaAufloesungC)))
print("AF = " + str(round(aufloesungF)) + "+-" + str(round(deltaAufloesungF)))

print(np.sqrt(A2 / (A0 - 1)))

print("lambda1 = " + str(np.sqrt(- A2 / (A0 - 1.))) + "+-" + str(.5 * np.sqrt(- A2 / (A0 - 1.) ** 3 * deltaA0 ** 2 + 1. / (- A2 * (A0 - 1.)) * deltaA2 ** 2)))


