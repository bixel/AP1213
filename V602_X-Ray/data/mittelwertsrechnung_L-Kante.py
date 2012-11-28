import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

rydberg = 13.605692
alpha = 1. / 137.
print(alpha)
h = 6.62606876e-34
e = 1.602176462e-19
c = 299792458
d = 201e-12
deg_to_rad = 2 * np.pi / 360

# z = 80
# x_1 = np.array([45.2, 45.5, 46.0])
# x_2 = np.array([40.8, 41.2, 41.5])

# z = 79
# x_1 = np.array([45.2, 45.5, 46.0])
# x_2 = np.array([40.4, 40.8, 41.2, 41.5])

z = 29
x_1 = np.array([45.2, 45.5, 46.0])
x_2 = np.array([40.8, 41.2, 41.5])

x_1 /= 2
x_2 /= 2

x_1_quer = 0
for val in x_1:
	x_1_quer += val

x_1_quer /= x_1.size

# Fehler auf E1
x_1_err = (((h * c) / (2 * d)) * (np.cos(x_1_quer * deg_to_rad) / (np.sin(x_1_quer * deg_to_rad) ** 2)) * (0.1 * deg_to_rad)) / e

x_2_quer = 0
for val in x_2:
	x_2_quer += val

x_2_quer /= x_2.size

# Fehler auf E2
x_2_err = (((h * c) / (2 * d)) * (np.cos(x_2_quer * deg_to_rad) / (np.sin(x_2_quer * deg_to_rad) ** 2)) * (0.1 * deg_to_rad)) / e

# in Energie
x_1 = h * c / (2 * d * np.sin(x_1 * deg_to_rad)) / e
x_2 = h * c / (2 * d * np.sin(x_2 * deg_to_rad)) / e

x_1_sum = 0
for val in x_1:
	x_1_sum += val

x_1_sum /= x_1.size

x_2_sum = 0
for val in x_2:
	x_2_sum += val

x_2_sum /= x_2.size

print('E1: ' + str(x_1_sum))
print('E2: ' + str(x_2_sum))

if x_2_sum > x_1_sum:
	x_delta = x_2_sum - x_1_sum
else:
	x_delta = x_1_sum - x_2_sum

#x_delta = 1814.9		

sig = z - np.sqrt((4 / alpha * np.sqrt(x_delta / rydberg) - 5 * x_delta / rydberg) * (1 + 19/32 * (alpha ** 2) * x_delta / rydberg))
#sig = 0
sig_err = sig * ((x_1_err + x_2_err) / x_delta)
sig_error_krass = (- 95 * (alpha ** 3) * x_delta ** (3 / 2) + 57 * (alpha ** 2) * np.sqrt(rydberg) * x_delta - 80 * alpha * rydberg * np.sqrt(x_delta) + 32 * (rydberg ** (3/2)))/(4 * np.sqrt(2) * alpha * (rydberg ** 2) * np.sqrt(x_delta) * np.sqrt((4 * np.sqrt(x_delta))/(alpha * np.sqrt(rydberg)) - (5 * x_delta)/rydberg) * np.sqrt((19 * (alpha ** 2) * x_delta)/rydberg + 32)) * np.sqrt(x_1_err ** 2 + x_2_err ** 2)

print('Mittelwert w1: ' + str(x_1_quer))
print('Mittelwert w2: ' + str(x_2_quer))
print('E_delta: ' + str(x_delta))
print('Fehler auf E_1_Mittel: ' + str(x_1_err))
print('Fehler auf E_2_Mittel: ' + str(x_2_err))
print('simga: ' + str(sig))
print('Fehler auf sigma: ' + str(sig_err))
print('Krasser Fehler auf sigma: ' + str(sig_error_krass))


