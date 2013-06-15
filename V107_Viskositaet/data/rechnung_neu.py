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

#kugel klein
t_klein = np.genfromtxt("kugel_gross.txt", unpack = True)
summe = t_klein.sum()
mittel = summe / t_klein.size
t_klein -= mittel
error = 0
for val in t_klein:
	error += val**2
error = np.sqrt(error / (t_klein.size * (t_klein.size - 1)))
print("fehler t: " + str(error))

print("fehler dichte_k: " + str(6 / (np.pi * 1.582**3) * ( 3 * 4.6 * .05 / 15.82 + .05)))

print("fehler eta: " + str(K * ((mittel * error) + (2.223 - .998) * .04)))

d = 1.221
t = 95.2
eta = 1.196
d_eta = .006
d_rho = .045
d_t = .13
print("fehler K: " + str(1000 * (d_eta / (d * t) + eta / (d * t) *( d_rho / d + d_t / t))))

print(np.log(np.e))

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
