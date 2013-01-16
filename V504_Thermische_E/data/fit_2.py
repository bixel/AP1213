import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

U, I = np.genfromtxt("messung_2.txt", unpack = True)
U -= I
for val in U:
	print(val)

fickifacki = lambda x, a, b: a * x + b

p, cov = curve_fit(fickifacki, U, np.log(I), maxfev = 1000)
print("c = " + str(p[0]))
print("c_err = " + str(np.sqrt(cov[0][0])))

plt.plot(U, fickifacki(U, p[0], p[1]), "r-")
plt.plot(U, np.log(I), "kx")

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 5)

plt.grid(which = 'both')
plt.savefig('../img/messung2.pdf', bbox_inches='tight')
plt.clf()

k = 1.3806503e-23
h = 6.62606876e-34
e = 1.602176462e-19
m = 9.10938188e-31

I = np.array([4e-9, 40e-4, 320e-9])
T = np.array([1741, 1894, 2033])
summe = 0
E = -k * T * np.log((I * (h**3)) / (4 * np.pi * e * m * k**2 * T**2))

print(np.sqrt(((E - E.sum()/3)**2).sum()/6) / e)