import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

# Konstanten

rydberg = 13.605692
alpha = 1/137
h = 6.62606876e-34
e = 1.602176462e-19
c = 299792458
d = 201e-12
deg_to_rad = 2 * np.pi / 360

# z = 32
# x = np.array([16.7, 17.2])

z = 41
x = np.array([18.0, 18.4, 18.7, 19.2, 19.6, 22.0])

x /= 2

x_quer = 0
for val in x:
	x_quer += val

x_quer /= x.size

x_err = (((h * c) / (2 * d)) * (np.cos(x_quer * deg_to_rad) / (np.sin(x_quer * deg_to_rad) ** 2)) * (0.1 * deg_to_rad)) / e

# Zuerst auf Energie umrechnen
x = h * c / (2 * d * np.sin(x * deg_to_rad)) / e

x_sum = 0
for val in x:
	x_sum += val

x_sum /= x.size.sin(x_sum * (2 * np.pi / 360))) / e

sig = z - np.sqrt((x_sum / rydberg - (alpha ** 2) * (z ** 2 / 4)))

sig_err = (1 / (2 * rydberg) / np.sqrt((x_sum) / rydberg - .25 * (z * alpha) ** 2) * (x_err))

print('Mittelwert Winkel: ' + str(x_quer))
print('E_Mittel: ' + str(x_sum))
print('Fehler auf E_Mittel: ' + str(x_err))
print('simga: ' + str(sig))
print('Fehler auf sigma: ' + str(sig_err))