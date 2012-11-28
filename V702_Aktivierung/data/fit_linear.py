import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

data = 'indium'

if data == 'rhodium':
	# Graph fuer Rhodium
	x, y = np.genfromtxt('Rhodium_12s.txt', unpack = True)

	x *= 12

	N_Dunkel = 256. / 900. * 12

	y_splitted = np.split(y, [18, 26, 40])
	x_splitted = np.split(x, [18, 26, 40])

	y1 = y_splitted[0]
	x1 = x_splitted[0]

	y2 = y_splitted[2]
	x2 = x_splitted[2]

	y1_error_arr = np.array([np.log(y1) - np.log(y1 - np.sqrt(y1 + N_Dunkel)), np.log(y1 + np.sqrt(y1 + N_Dunkel)) - np.log(y1)])
	y2_error_arr = np.array([np.log(y2) - np.log(y2 - np.sqrt(y2 + N_Dunkel)), np.log(y2 + np.sqrt(y2 + N_Dunkel)) - np.log(y2)])

	y -= N_Dunkel
	y1 -= N_Dunkel
	y2 -= N_Dunkel

	y = np.log(y)
	#plt.errorbar(x, y, yerr = 1 / np.sqrt(y ** 2 + N_Dunkel), fmt = 'kx')

	x1_theorie = np.arange(0, 230.1, .2)
	x1_theorie_punkte = np.arange(230, 395.1, .2)
	x2_theorie = np.arange(310, 500.1, .2)
	x2_theorie_punkte = np.arange(0, 310.1, .2)

	m_vorschlag = .01
	n_0_vorschlag = 3000

	funk = lambda t, m, n_0: np.log(n_0 * (1 - np.e ** (-m * 12))) - m * t

	# Zuerst fuer grosse t
	y2 = np.log(y2)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p2, cov2 = curve_fit(funk, x2, y2, p0 = vor_werte, maxfev = 1000)
	print('Zweiter Teil: ')
	print(p2)
	print(np.sqrt(cov2[0][0]))

	# danach fuer kleine t: lambda_l muss dann herausgerechnet werden
	y1 -= p2[1] * (1 - np.e ** (- p2[0] * 12)) * np.e ** (- p2[0] * x1)
	y1 = np.log(y1)

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])
	p, cov = curve_fit(funk, x1, y1, p0 = vor_werte, maxfev = 1000)
	print('Erster Teil: ')
	print(p)
	print(np.sqrt(cov[0][0]))

	# m_vorschlag = .001
	# n_0_vorschlag = 1.0

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{logarithmierte \, Zerf\"alle}$')

	gerade_1, = plt.plot(x1_theorie, funk(x1_theorie, p[0], p[1]), 'r-')
	plt.plot(x1_theorie_punkte, funk(x1_theorie_punkte, p[0], p[1]), 'r--')
	werte_1 = plt.errorbar(x1, y1, yerr = y1_error_arr, fmt = 'gx')

	gerade_2, = plt.plot(x2_theorie, funk(x2_theorie, p2[0], p2[1]), 'b-')
	plt.plot(x2_theorie_punkte, funk(x2_theorie_punkte, p2[0], p2[1]), 'b--')
	werte_2 = plt.errorbar(x2, y2, yerr = y2_error_arr, fmt = 'yx')

	plt.legend((werte_1, gerade_1, werte_2, gerade_2), (r'$\ln{\left(N_{\Delta t} - N_{\Delta , l} - N_0 \right)}$', r'$\mathrm{Regressionsgerade\ 1}$', r'$\ln{\left( N_{\Delta t} - N_0 \right)}$', r'$\mathrm{Regressionsgerade\ 2}$'))

	t_halb1 = np.log(2) / p[0]
	t_halb2 = np.log(2) / p2[0]
	t_halb1_error = np.log(2) * np.sqrt(cov[0][0]) / (p[0] ** 2)
	t_halb2_error = np.log(2) * np.sqrt(cov2[0][0]) / (p2[0] ** 2)

	print('=> Halbwertszeiten 1 und 2: ' + str(t_halb1) + ' s1 = ' + str(dt.timedelta(seconds = t_halb1)) + '; ' + str(t_halb2) + ' s2 = ' + str(dt.timedelta(seconds = t_halb2)))
	print('=> Fehler 1: ' + str(t_halb1_error) + ' = ' + str(t_halb1_error / t_halb1 * 100) + '%; Fehler 2: ' + str(t_halb2_error) + ' = ' + str(t_halb2_error / t_halb2 * 100) + '%')

	print(np.log(2))

elif data == 'indium':
	# Graph fuer Indium
	x, y = np.genfromtxt('Indium_240s.txt', unpack = True)

	x *= 240
	N_Dunkel = 256. / 900. * 240

	for val in y:
		print(round(val - N_Dunkel))

	y_error = np.sqrt(y)
	y_error_arr = np.array([np.log(y) - np.log(y - np.sqrt(y + N_Dunkel)), np.log(y + np.sqrt(y + N_Dunkel)) - np.log(y)])
	y -= N_Dunkel

	y = np.log(y)

	print(y_error_arr)

	x_theorie = np.arange(0, 4001, 1)

	plt.xlabel(r'$t\,[\mathrm{s}]$')
	plt.ylabel(r'$\mathrm{logarithmierte \, Zerf\"alle}$')

	m_vorschlag = np.log(2) / 60
	n_0_vorschlag = 50000

	vor_werte = np.array([m_vorschlag, n_0_vorschlag])

	funk = lambda t, m, n_0: np.log(n_0 * (1 - np.e ** (-m * 240))) - m * t

	p, cov = curve_fit(funk, x, y, p0 = vor_werte, maxfev = 1000)
	print(p)
	print(cov)

	gerade, = plt.plot(x_theorie, funk(x_theorie, p[0], p[1]), 'r-')
	werte = plt.errorbar(x, y, yerr = y_error_arr, fmt = 'kx')
	plt.legend((gerade, werte), (r'$\mathrm{Regressionsgerade}$', r'$\ln{\left( N_{\Delta t} - N_0 \right)}$'))

	t_halb = np.log(2) / p[0]
	t_halb_err = np.log(2) * np.sqrt(cov[0][0]) / (p[0] ** 2)
	print('=> lambda: ' + str(p[0]))
	print('=> Fehler lambda: ' + str(np.sqrt(cov[0][0])))
	print('=> N_0: ' + str(p[1]))
	print('=> Halbwertszeit: ' + str(t_halb) + ' s = ' + str(dt.timedelta(seconds = t_halb)))
	print('=> Fehler T_h: ' + str(t_halb_err) + ' = ' + str(t_halb_err / t_halb * 100) + '%')

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/graph_' + data + '_linearisiert.pdf', bbox_inches='tight')
plt.clf()