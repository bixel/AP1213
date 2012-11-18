import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt

data = 'rhodium'

if data == 'rhodium':
	# Graph fuer Rhodium
	x, y = np.genfromtxt('Rhodium_12s.txt', unpack = True)

	x *= 12

	N_Dunkel = 256 / 900 * 12
	y -= N_Dunkel

	y_splitted = np.split(y, [19, 21, 40])
	x_splitted = np.split(x, [19, 21, 40])

	y1 = y_splitted[0]
	x1 = x_splitted[0]

	y2 = y_splitted[2]
	x2 = x_splitted[2]

	print(y2)

	x_theorie = np.arange(0, 500.1, .2)

	m_vorschlag = .001
	n_0_vorschlag = 1.0

	funk = lambda t, m, n_0: n_0 * (1 - np.e ** (-m * 12)) * np.e**(- m * t)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p2, cov2 = curve_fit(funk, x2, y2, p0 = vor_werte, maxfev = 1000)
	print('Zweiter Teil: ')
	print(p2)

	y1 -= p2[1] * (1 - np.e ** (- p2[0] * 12)) * np.e ** (- p2[0] * x1)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p, cov = curve_fit(funk, x1, y1, p0 = vor_werte, maxfev = 1000)
	print('Erster Teil: ')
	print(p)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[1 / 12\mathrm{s}]$')

	plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')
	plt.errorbar(x1, y1, yerr = np.sqrt(y1), fmt = 'gx')

	plt.plot(x_theorie, funk(x_theorie, p2[0], p2[1]), 'b-')
	plt.errorbar(x2, y2, yerr = np.sqrt(y2), fmt = 'yx')

	t_halb1 = np.log(2) / p[0]
	t_halb2 = np.log(2) / p2[0]

	print('=> Halbwertszeiten 1 und 2: ' + str(t_halb1) + 's1 = ' + str(dt.timedelta(seconds = t_halb1)) + '; ' + str(t_halb2) + 's2 = ' + str(dt.timedelta(seconds = t_halb2)))

elif data == 'indium':
	# Graph fuer Indium
	x, y = np.genfromtxt('Indium_240s.txt', unpack = True)

	x *= 240

	y -= 256 / 900 * 240

	y_error = np.sqrt(y)
	#y_error = np.log(y)
	#y = np.log(y)

	x_theorie = np.arange(0, 15 * 240.001, 1)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[1 / 240\mathrm{s}]$')

	m_vorschlag = 2.127e-4
	n_0_vorschlag = 1.0
	vor_werte = np.array([m_vorschlag, n_0_vorschlag])

	funk = lambda t, m, n_0: n_0 * (1 - np.e ** (-m * 240)) * np.e**(- m * t)

	p, cov = curve_fit(funk, x, y, p0 = vor_werte, maxfev = 5000)
	print('Erster Teil: ')
	print(p)
	print(cov)
	plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')

	plt.errorbar(x, y, yerr = np.sqrt((y_error ** 2) + 256 / 900 * 240), fmt = 'kx')

	t_halb = np.log(2) / p[0]

	print('Halbwertszeit: ' + str(t_halb) + 's = ' + str(dt.timedelta(seconds = t_halb)))

plt.grid(which = 'both')
plt.savefig('../img/graph_' + data + '.png')
plt.clf()