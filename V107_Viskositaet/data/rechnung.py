import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

x, y = np.genfromtxt("temperatur.txt", unpack = True)
#x = np.arange(1, 10, 1)

print(x, y)

y_mittel = np.sum(y) / y.size
y_error = np.sqrt(10 * .1 ** 2)

counter = 0
wert = 0
for val in y:
	wert += val
	if counter == 0:
		counter += 1
	else:
		counter = 0
		print(wert / 2.)
		wert = 0

print(y_mittel)
print(y_error)