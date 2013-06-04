import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

r, F, phi = np.genfromtxt('messung1_phi.txt', unpack = True)

r *= .01
F *= .001
phi *= np.pi / 180.

D = F * r / phi

# for i in range(0, np.size(D)):
# 	print(np.round(D[i] * 1000, 2))
# 	print(phi[i] * 180. / np.pi)

D_ges = np.sum(D) / np.size(D)
D_err = np.sqrt(1. / (np.size(D) - 1.) * np.sum((D - D_ges)**2))

print(D_ges)
print(D_err)

#D_ges = 0.0579108790623 # Aus Ausgleichsrechnung

T, N = np.genfromtxt('messung6_puppe-gross.txt', unpack = True)

# for i in range(0, np.size(T)):
# 	print(np.round(T[i], 2))

T /= 3.

# for i in range(0, np.size(T)):
# 	print(np.round(T[i], 2))

T_ges = np.sum(T) / np.size(T)
T_err = np.sqrt(1. / (np.size(T) - 1) * np.sum((T - T_ges)**2))

I_ges = T_ges**2 * D_ges / (4. * np.pi**2)
I_err = np.sqrt((T_ges*D_ges / (2. * np.pi**2) * T_err) ** 2 + (T_ges**2 / (4. * np.pi**2) * D_err))

print("I_ges = " + str(I_ges * 1000) + "\nI_err = " + str(I_err))

Ik = 25.2
DelIk = 9.3
It = 438.8
DelIt = 155.3
Ia = 646.5
DelIa = 24.6
Ib = 27.9
DelIb = 9.6
a = 2.26 + 17.0 / 2
b = 3.
dela = .05
delb = .05
ma = 26.7
mb = 50.7
delma = .1
delmb = .1

print(Ik + It + 2 * Ia + 2 * Ib + 2 * ma * a**2 + 2 * mb * b**2)
print(np.sqrt(DelIk**2 + DelIt**2 + (2*DelIa)**2 + (2*DelIb)**2 + (2*a**2 *delma)**2 + (4*a*ma*dela)**2 + (2*b**2 * delmb)**2 + (4*b*mb*delb)**2))
