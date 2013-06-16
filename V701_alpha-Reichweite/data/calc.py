import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import poisson
import datetime as dt
import pylab
from uncertainties import ufloat
from uncertainties import unumpy

T = 120
l0 = 2.6
p0 = 1013


if True:
	for messung in [1,2]:
		print("\nMessung 1-" + str(messung) + ":")
		p, counts, maxCnl = np.genfromtxt("messung1-" + str(messung) + ".txt", unpack = True)

		lEff = l0 * p / p0
		zaehlrate = counts / T

		energieFaktor = 4 / maxCnl[0]
		energie = energieFaktor * maxCnl

		#Zaehlrate
		fitFunktion = lambda x, m, b: m * x + b
		vorschlagsWerte = np.array([-1., 1.])

		fitValueMin = 12
		fitValueMax = 15

		koeffizienten, varianz = curve_fit(fitFunktion, lEff[fitValueMin:fitValueMax], zaehlrate[fitValueMin:fitValueMax], p0 = vorschlagsWerte, maxfev = 1000)
		m = koeffizienten[0]
		b = koeffizienten[1]
		deltaM = np.sqrt(varianz[0][0])
		deltaB = np.sqrt(varianz[1][1])
		zaehlrateMax = np.max(zaehlrate)
		a = zaehlrateMax / 2

		xMittel = ((a - b) / m)
		deltaXMittel = np.sqrt((deltaB / m)**2 + ((a-b) * deltaM / m**2)**2)
		print("m = " + str(m) + "+-" + str(deltaM))
		print("b = " + str(b) + "+-" + str(deltaB))
		print("xMittel = " + str(xMittel) + "+-" + str(deltaXMittel))

		lEffTheorie = np.arange(0, np.max(lEff), 1e-6)
		plt.plot(lEffTheorie, fitFunktion(lEffTheorie, m, b), "b-")
		plt.plot(lEff, np.max(zaehlrate) / 2 + 0 * lEff, "g-")
		plt.plot(lEff, zaehlrate, "rx")
		plt.ylim([0, 600])
		plt.xlabel(r"$x_\mathrm{eff} \, [\mathrm{cm}]$")
		plt.ylabel(r"$\mathrm{Z\"ahlrate} \, [1 / \mathrm{s}]$")
		plt.legend(["Ausgleichsgerade", "Mittlere Zaehlrate", "Messwerte"], "lower left")

		fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
		fig.set_size_inches(10 * .7, 7 * .7)

		plt.grid(which = 'both')
		plt.savefig('../img/zaehlrate' + str(messung) + '.pdf')
		plt.clf()


		#Energie
		fitValueMin = 0
		fitValueMax = 11
		koeffizienten, varianz = curve_fit(fitFunktion, lEff[fitValueMin:fitValueMax], energie[fitValueMin:fitValueMax])
		m = koeffizienten[0]
		b = koeffizienten[1]
		deltaM = np.sqrt(varianz[0][0])
		deltaB = np.sqrt(varianz[1][1])

		print("energieXMittel = " + str(fitFunktion(xMittel, m, b)))
		print("dE/dx = " + str(m) + "+-" + str(deltaM))

		plt.plot(lEffTheorie, fitFunktion(lEffTheorie, m, b), "b-")
		plt.plot(lEff, energie, "rx")
		plt.xlabel(r"$x_\mathrm{eff} \, [\mathrm{cm}]$")
		plt.ylabel(r"$E \, [\mathrm{MeV}]$")
		plt.legend(["Ausgleichsgerade", "Messwerte"], "upper right")

		fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"

		plt.grid(which = 'both')
		plt.savefig("../img/energie" + str(messung) + ".pdf")
		plt.clf()


#Gauss- Posson- Kurven
print("Messung 2: Gauss- und Poissonkurve \n")
T = 10
counts = np.genfromtxt("messung2.txt", unpack = True)
zaehlrate = counts / T

mittelwert = np.sum(zaehlrate) / np.size(zaehlrate)
varianz = 1. / (np.size(zaehlrate) - 1.) * np.sum((zaehlrate - mittelwert) ** 2)

print("zaehlrateMittel = " + str(round(mittelwert, 3)))
print("varianz = " + str(round(varianz, 3)))

balkenzahl = 8
breite = (np.max(zaehlrate) - np.min(zaehlrate)) / balkenzahl
balken = np.array([])

for balkenNummer in range(1, balkenzahl + 1, 1):
	untereSchranke = np.min(zaehlrate) + (balkenNummer - 1) * breite
	obereSchranke = np.min(zaehlrate) + balkenNummer * breite
	balken = np.append(balken, [0])
	for i in range(0, np.size(zaehlrate), 1):
		if (zaehlrate[i] >= untereSchranke) & (zaehlrate[i] < obereSchranke):
			balken[balkenNummer - 1] += 1
	print("Bereich " + str(balkenNummer) + ": \t" + str(round(untereSchranke, 3)) + "\t bis \t" + str(round(obereSchranke, 3)) + "\t: " + str(balken[balkenNummer - 1]))

k = np.arange(0, balkenzahl, 1)

x = np.arange(-2, 10, .001)
varianz = 2
mittelwert = 4.5
gaussFunktion = lambda x, A: A / (varianz * np.sqrt(2 * np.pi)) * np.e ** (- .5 * ((x - mittelwert) / varianz) ** 2)

koeffizienten, unsicherheit = curve_fit(gaussFunktion, balkenNummer, balken, maxfev = 1000)

fig = plt.gcf()

plt.bar(k + 1, balken / np.sum(balken), color = "r", width = .8)
plt.plot(x, gaussFunktion(x, 1), "b-")
#plt.hist(zaehlrate, balkenzahl, color="r", normed = True, align = "mid")

plt.grid(which = "both")
plt.xlim([0, 10])
plt.ylim([0, .25])
plt.xlabel(r"$\mathrm{H\"aufigkeitsbereich}$")
plt.ylabel(r"$\mathrm{rel.\, H\"aufigkeit}$")
plt.legend(["Messwerte", "Gaussfunktion"], "upper right")
plt.savefig("../img/verteilungGauss.pdf")
plt.clf()

poissonLambda = 4.
poissonBalken = np.array([])
kPoisson = np.arange(0, balkenzahl + 2, 1)
for i in kPoisson:
	poissonBalken = np.append(poissonBalken, [poissonLambda ** i / math.factorial(i) * math.exp(-poissonLambda)])

fig = plt.gcf()

plt.bar(k + 1, balken / np.sum(balken), color = "r", width = .4)
plt.bar(kPoisson + .4, poissonBalken, color = "b", width = .4)

plt.grid(which = "both")
plt.legend(["Messwete", "Poissonverteilung"], "upper right")
plt.ylim([0, .25])
plt.xlabel(r"$\mathrm{H\"aufigkeitsbereich}$")
plt.ylabel(r"$\mathrm{rel.\, H\"aufigkeit}$")
plt.savefig("../img/verteilungPoisson.pdf")
plt.clf()