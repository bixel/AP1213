import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

# Benoetigte Konstanten definieren
K = 10.3e-3
delta_K = .6e-3
p_k = 2.219
delta_p_k = .066
p_fl = .998
delta_t = np.sqrt(.002)

# Messdaten einlesen
x, y = np.genfromtxt("temperatur.txt", unpack = True)

# Gesamtmittelwert und Fehler (Wird nur bei ersten beiden Messungen benoetigt)
# y_mittel = np.sum(y) / y.size
# y_error = np.sqrt(10 * .1 ** 2)

# Temperaturwerte und Werte fuer Theoriekurve
x_arr = np.arange(27, 54.1, 3)
x_theorie = np.arange(27, 54.001, .1)

# Array fuer Einzelne Mittelwerte (t_quer_2) erstellen und fuellen
y_mittel_arr = np.array([])
counter = 0
wert = 0
for val in y:
	wert += val
	if counter == 0:
		counter += 1
	else:
		counter = 0
		wert /= 2.
		#print(wert)
		# Jeweilige Viskositaet an das array anfuegen
		y_mittel_arr = np.append(y_mittel_arr, [(p_k - p_fl) * K * wert], axis = 0)
		wert = 0

print(min(y_mittel_arr))

# Fehler der eta-Werte
y_mittel_arr_err = (p_k - p_fl) * y_mittel_arr * delta_K + K * y_mittel_arr * delta_p_k + K * (p_k - p_fl) * delta_t

# Linearisierte Werte plotten
plt.plot(1 / x_arr, np.log(y_mittel_arr), 'kx')

# Linearer Fit
vorschlag = np.array([1, 1])
function = lambda t, m, b: m * t + b
p, cov = curve_fit(function, 1 / x_arr, np.log(y_mittel_arr), p0 = vorschlag, maxfev = 1000)

# Fit plotten (linear)
plt.plot(1 / x_theorie, function(1 / x_theorie, p[0], p[1]), 'k-')

print(p)
print(cov)

# Koeffizienten leserlich ausgeben
print('B = ' + str(p[0]) + ' +- ' + str(np.sqrt(cov[0][0])))
print('A = ' + str(np.e ** p[1]) + ' +- ' + str(np.sqrt(cov[1][1])))

# Plot Einstellungen
plt.grid(which = 'major', axis = 'both')
plt.ylabel(r'$\eta \quad (\mathrm{linearisiert})$')
plt.xlabel(r'$1 / T \quad [1 / \degree \mathrm{C}]$')

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(8, 4)

plt.savefig('../img/plot.pdf', bbox_inches = 'tight')
plt.clf()

# nichtlinearisierte Werte plotten
plt.plot(x_arr, y_mittel_arr, 'kx')
# plt.errorbar(x_arr, y_mittel_arr, yerr = y_mittel_arr_err, fmt = 'rx')
plt.plot(x_theorie, np.e ** p[1] * (np.e ** (p[0] / x_theorie)), 'k-')

# Plot Einstellungen
plt.grid(which = 'major', axis = 'both')
plt.ylabel(r'$\eta \quad [\mathrm{kg\,m^{-1}\,s^{-1}}]$')
plt.xlabel(r'$T \quad [\degree \mathrm{C}]$')

plt.savefig('../img/plot2.pdf', bbox_inches = 'tight')
plt.clf()