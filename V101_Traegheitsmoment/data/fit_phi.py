import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

r, F, phi = np.genfromtxt('messung1_phi.txt', unpack = True)

r *= .01
F *= .001
phi *= (2 * np.pi) / 360

D = F * r / phi

for i in range(0, np.size(D)):
	print(np.round(D[i] * 1000, 2))

D_ges = np.sum(D) / np.size(D)
D_err = np.sqrt(1. / (np.size(D) - 1.) * np.sum((D - D_ges)**2))

print(D_ges)
print(D_err)

D_ges = 0.0579108790623

T, N = np.genfromtxt('messung3_messingzylinder.txt', unpack = True)

for i in range(0, np.size(T)):
	print(np.round(T[i], 2))

T /= 3.

for i in range(0, np.size(T)):
	print(np.round(T[i], 2))

T_ges = np.sum(T) / np.size(T)
T_err = np.sqrt(1. / (np.size(T) - 1) * np.sum((T - T_ges)**2))

I_ges = T_ges**2 * D_ges / (4. * np.pi**2)
I_err = np.sqrt((T_ges*D_ges / (2. * np.pi**2) * T_err) ** 2 + (T_ges**2 / (4. * np.pi**2) * D_err))

print("I_ges = " + str(I_ges * 1000) + "\nI_err = " + str(I_err))

Ik = 9.3
It = 155.3
Ia = 24.6
Ib = 9.6
a = 8.5
b = 3.
dela = .05
delb = .05
ma = 26.7
mb = 50.7
delma = .1
delmb = .1

print("OMG")
print(np.sqrt(Ik**2 + It**2 + (2*Ia)**2 + (2*Ib)**2 + (2*a**2 *delma)**2 + (4*a*ma*dela)**2 + (2*b**2 * delmb)**2 + (4*b*mb*delb)**2))
