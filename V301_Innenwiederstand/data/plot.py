import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

# Aufgabe a - Plot U gegen I fuer alle Spannungsquellen

messungsArray = np.array(["monozelle", "sinus", "rechteck"])

for messung in messungsArray:
	I, U = np.genfromtxt("messung_" + messung + ".txt", unpack = True)

	# for i in range(0, np.size(I)):
	# 	print(str(np.round(U[i] * I[i], 3)) + " +- " + str(np.round(np.sqrt((I[i] * U[i] * .02) ** 2 + (I[i] * U[i] * .015) ** 2), 0)))

	if messung == "monozelle":
		plt.ylabel(r"$U \, [\mathrm{V}]$")
		plt.xlabel(r"$I \, [\mathrm{mA}]$")
	elif messung == "sinus":
		plt.ylabel(r"$U \, [\mathrm{V}]$")
		plt.xlabel(r"$I \, [\mathrm{mA}]$")
	elif messung == "rechteck":
		plt.ylabel(r"$U \, [\mathrm{mV}]$")
		plt.xlabel(r"$I \, [\mathrm{mA}]$")

	Itheorie = np.arange(0, np.max(I), .00001)
	
	plt.errorbar(I, U, xerr = .015 * I, yerr = .02 * U, fmt = "ko")

	fittingFunction = lambda i, u0, ri: u0 - i * ri

	koeffizienten, varianz = curve_fit(fittingFunction, I, U, maxfev = 1000)

	plt.plot(Itheorie, fittingFunction(Itheorie, koeffizienten[0], koeffizienten[1]), "r-")

	if messung == "monozelle":
		I2, U2 = np.genfromtxt("messung_c.txt", unpack = True)
		fittingFunction2 = lambda i, u0, ri: u0 + i * ri
		I2theorie = np.arange(0, np.max(I2), .00001)

		koeffizienten2, varianz2 = curve_fit(fittingFunction2, I2, U2, maxfev = 1000)

		plt.errorbar(I2, U2, xerr = .015 * I2, yerr = .02 * U2, fmt = "ko")
		plt.plot(I2theorie, fittingFunction2(I2theorie, koeffizienten2[0], koeffizienten2[1]), "b-")

		print("U0gegen = " + str(np.round(koeffizienten2[0], 5)) + " +- " + str(np.round(np.sqrt(varianz2[0][0]), 5)))
		print("Rigegen = " + str(np.round(koeffizienten2[1], 5)) + " +- " + str(np.round(np.sqrt(varianz2[1][1]), 5)) + "\n")
		plt.legend(np.array(["Messwerte", "Ausgleichsgerade"]), "upper left")
	else:
		plt.legend(np.array(["Messwerte", "Ausgleichsgerade"]))

	print("U0_" + messung + " = " + str(np.round(koeffizienten[0], 5)) + " +- " + str(np.round(np.sqrt(varianz[0][0]), 5)))
	print("Ri_" + messung + " = " + str(np.round(koeffizienten[1], 5)) + " +- " + str(np.round(np.sqrt(varianz[1][1]), 5)) + "\n")

	fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
	fig.set_size_inches(19 * 0.393700787, 11 * 0.393700787)

	plt.grid(which = 'both')
	plt.savefig('../img/graph_' + messung + '.pdf')
	plt.clf()

	if messung == "monozelle":
		I /= 1000
		koeffizienten[1] *= 1000
		R = U / I

		Rerror = np.sqrt((U / I**2 * I * .015) ** 2 + (1 / I * U * .02) ** 2)
		P = U * I
		Perror = np.sqrt((I * U * .02) ** 2 + (I * U * .015) ** 2)

		Istep = (np.max(I) - np.min(I)) / 1000
		Itheorie = np.arange(np.min(I), np.max(I), Istep)
		Rstep = (np.max(R) - np.min(R)) / 1000
		Rtheorie = np.arange(np.min(R), np.max(R), Rstep)
		Rtheorie = Rtheorie[::-1]

		plotFunction = lambda ra: (koeffizienten[0] ** 2 * ra) / (ra + koeffizienten[1]) ** 2

		plt.errorbar(R, P, xerr = Rerror, yerr = Perror, fmt = "ko")
		plt.plot(Rtheorie, plotFunction(Rtheorie), "r-")

		plt.xlabel(r"$R \, [\Omega]$")
		plt.ylabel(r"$P \, [\mathrm{W}]$")
		plt.legend(np.array(["Messpunkte", "Theoriekurve"]), "lower right")

		fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
		fig.set_size_inches(19 * 0.393700787, 11 * 0.393700787)

		plt.grid(which = 'both')
		plt.savefig('../img/graph_' + messung + '_leistung.pdf')
		plt.clf()