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

	y = np.log(y)
	#plt.errorbar(x, y, yerr = 1 / np.sqrt(y ** 2 + N_Dunkel), fmt = 'kx')

	x1_theorie = np.arange(0, 260.1, .2)
	x1_theorie_punkte = np.arange(260, 370.1, .2)
	x2_theorie = np.arange(300, 500.1, .2)
	x2_theorie_punkte = np.arange(0, 300.1, .2)

	m_vorschlag = .001
	n_0_vorschlag = 1000

	funk = lambda t, m, n_0: np.log(n_0 * (1 - np.e ** (-m * 12))) - m * t

	# Zuerst fuer grosse t
	y2 = np.log(y2)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p2, cov2 = curve_fit(funk, x2, y2, p0 = vor_werte, maxfev = 1000)
	print('Zweiter Teil: ')
	print(p2)

	# danach fuer kleine t: lambda_l muss dann herausgerechnet werden
	y1 -= p2[1] * (1 - np.e ** (- p2[0] * 12)) * np.e ** (- p2[0] * x1)
	y1 = np.log(y1)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p, cov = curve_fit(funk, x1, y1, p0 = vor_werte, maxfev = 1000)
	print('Erster Teil: ')
	print(p)

	# m_vorschlag = .001
	# n_0_vorschlag = 1.0

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[\mathrm{ln}(1 / 12\mathrm{s})]$')

	gerade_1, = plt.plot(x1_theorie, funk(x1_theorie, p[0], p[1]), 'r-')
	plt.plot(x1_theorie_punkte, funk(x1_theorie_punkte, p[0], p[1]), 'r--')
	werte_1 = plt.errorbar(x1, y1, yerr = 1 / np.sqrt(y1 ** 2 + N_Dunkel), fmt = 'gx')

	gerade_2, = plt.plot(x2_theorie, funk(x2_theorie, p2[0], p2[1]), 'b-')
	plt.plot(x2_theorie_punkte, funk(x2_theorie_punkte, p2[0], p2[1]), 'b--')
	werte_2 = plt.errorbar(x2, y2, yerr = 1 / np.sqrt(y2 ** 2 + N_Dunkel), fmt = 'yx')

	plt.legend((werte_1, gerade_1, werte_2, gerade_2), (r'$\ln{\left(N_{\Delta t} - N_{\Delta , l} - N_0 \right)}$', r'$\mathrm{Theoriekurve\ 1}$', r'$\ln{\left( N_{\Delta t} - N_0 \right)}$', r'$\mathrm{Theoriekurve\ 2}$'))

	t_halb1 = np.log(2) / p[0]
	t_halb2 = np.log(2) / p2[0]

	print('=> Halbwertszeiten 1 und 2: ' + str(t_halb1) + 's1 = ' + str(dt.timedelta(seconds = t_halb1)) + '; ' + str(t_halb2) + 's2 = ' + str(dt.timedelta(seconds = t_halb2)))

elif data == 'indium':
	# Graph fuer Indium
	x, y = np.genfromtxt('Indium_240s.txt', unpack = True)

	x *= 240

	y -= 256 / 900 * 240
	y = np.log(y)

	y_error = np.sqrt(y)
	#y_error = np.log(y)
	#y = np.log(y)

	x_theorie = np.arange(0, 4001, 1)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{Zerf\"alle}\,[1 / 240\mathrm{s}]$')

	m_vorschlag = 2.127e-4
	n_0_vorschlag = 50000
	vor_werte = np.array([m_vorschlag, n_0_vorschlag])

	funk = lambda t, m, n_0: np.log(n_0 * (1 - np.e ** (-m * 240))) - m * t

	p, cov = curve_fit(funk, x, y, p0 = vor_werte, maxfev = 1000)
	print('Erster Teil: ')
	print(p)
	plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')

	plt.errorbar(x, y, yerr = 1 / np.sqrt((y_error ** 2) + 256 / 900 * 240), fmt = 'kx')

	t_halb = np.log(2) / p[0]
	print('=> lambda: ' + str(p[0]))
	print('=> N_0: ' + str(p[1]))
	print('=> Halbwertszeit: ' + str(t_halb) + 's = ' + str(dt.timedelta(seconds = t_halb)))

plt.grid(which = 'both')
plt.savefig('../img/graph_' + data + '_linearisiert.png')
plt.clf()