import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

e, g1, g2 = np.genfromtxt("bessel.txt", unpack = True)
b1 = e - g1
b2 = e - g2
d = (np.sqrt((g1 - b1)**2) + np.sqrt((g2 - b2)**2)) / 2
f = (e**2 - d**2) / (4 * e)

index = 0
while index < 10:
	print(g1[index], b1[index], g2[index], b2[index])
	index += 1

print("e:")
for i in range(0, d.size):
	print(str(round(e[i], 3)))

print("d:")
for i in range(0, d.size):
	print(str(round(d[i], 3)))

print("f:")
for i in range(0, d.size):
	print(str(round(f[i], 2)))

f_gesamt = f.sum() / f.size
print("f_quer = " + str(f_gesamt) + "mm")
f_fehler = np.sqrt(((f - f_gesamt) ** 2).sum() / (f.size * (f.size - 1)))
print("f_fehler = " + str(f_fehler) + "mm")

# Rot
e, g1, g2 = np.genfromtxt("bessel_rot.txt", unpack = True)
b1 = e - g1
b2 = e - g2
d = g1 - b1
f = (e**2 - d**2) / (4 * e)

index = 0
while index < g1.size:
	print(g1[index], b1[index], g2[index], b2[index])
	index += 1

f_gesamt = f.sum() / f.size
rot = f_gesamt
print("f_rot_quer = " + str(f_gesamt) + "mm")
f_fehler = np.sqrt(((f - f_gesamt) ** 2).sum() / (f.size * (f.size - 1)))
print("f_rot_fehler = " + str(f_fehler) + "mm")

# Blau
e, g1, g2 = np.genfromtxt("bessel_blau.txt", unpack = True)
b1 = e - g1
b2 = e - g2
d = g1 - b1
f = (e**2 - d**2) / (4 * e)
index = 0
while index < g1.size:
	print(g1[index], b1[index], g2[index], b2[index])
	index += 1

f_gesamt = f.sum() / f.size
blau = f_gesamt
print("f_blau_quer = " + str(f_gesamt) + "mm")
f_fehler = np.sqrt(((f - f_gesamt) ** 2).sum() / (f.size * (f.size - 1)))
print("f_blau_fehler = " + str(f_fehler) + "mm")

print("delta_f = " + str(rot - blau) + "mm")