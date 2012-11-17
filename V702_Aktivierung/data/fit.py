import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt('Rhodium_12s.txt', unpack = True)

data = 'rhodium'

if data == 'rhodium':
	# Graph fuer Rhodium
	x *= 12

	y -= 256 / 900 * 12

	y_splitted = np.split(y, 2)
	x_splitted = np.split(x, 2)
	y1 = y_splitted[0]
	x1 = x_splitted[0]

	y2 = y_splitted[1]
	x2 = x_splitted[1]

	print(y2)

	x_theorie = np.arange(0, 500.1, .2)

	m_vorschlag = .001
	n_0_vorschlag = 1.0

	funk = lambda t, m, n_0: n_0 * np.e**(- m * t)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p, cov = curve_fit(funk, x1, y1, p0 = vor_werte, maxfev = 1000)
	print('Erster Teil: ')
	print(p)

	m_vorschlag = .001
	n_0_vorschlag = 1.0

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p2, cov2 = curve_fit(funk, x2, y2, p0 = vor_werte, maxfev = 1000)
	print('Zweiter Teil: ')
	print(p2)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[1 / 12\mathrm{s}]$')

	plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')
	plt.errorbar(x1, y1, yerr = np.sqrt(y1), fmt = 'kx')

	plt.plot(x_theorie, funk(x_theorie, p2[0], p2[1]), 'b-')
	plt.errorbar(x2, y2, yerr = np.sqrt(y2), fmt = 'kx')

elif data == 'indium':
	# Graph fuer Indium
	x *= 240

	y -= 256 / 900 * 240

	y_error = np.sqrt(y)
	#y_error = np.log(y)
	#y = np.log(y)

	x_theorie = np.arange(0, 15*240.001, 1)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[1 / 240\mathrm{s}]$')

	m_vorschlag = .01
	n_0_vorschlag = 1.0
	vor_werte = np.array([m_vorschlag, n_0_vorschlag])

	funk = lambda t, m, n_0: n_0 * np.e**(- m * t)

	p, cov = curve_fit(funk, x, y, p0 = vor_werte, maxfev = 5000)
	print('Erster Teil: ')
	print(p)
	print(cov)
	plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')
	plt.errorbar(x, y, yerr = y_error, fmt = 'x')

plt.savefig('../img/graph_indium.png')
plt.clf()