import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

I_H, U_H, I_raw, U_raw = np.genfromtxt("messung_1.txt", unpack = True)
I, U = [], [] 

index = 1
start = 0
for i in range(5):
	while ((index < len(I_H)) and I_H[index] == I_H[index - 1]):
		index += 1
	I.append(I_raw[start:index])
	U.append(U_raw[start:index])
	start = index
	index += 1 #Korrektur wegen Leerzeile zwischen Messreihen

# plt.plot(U[0], I[0], "rx")
# plt.plot(U[0], .004+ 0 * U[0], "r-")
# plt.plot(U[1], I[1], "bx")
# plt.plot(U[1], .04+ 0 * U[1], "b-")
# plt.plot(U[2], I[2], "cx")
# plt.plot(U[2], .32+ 0 * U[2], "c-")
# plt.plot(U[3], I[3], "mx")
# plt.plot(U[4], I[4], "gx")

# plt.plot(U[1], 1.25 + 0 * U[1], "r--")
# plt.annotate(r"$\mathrm{\"Uberlast\,des\,Strommessger\"ates}$", (130, 1.27), xycoords = "data")
# plt.xlabel(r"$U\,[\mathrm{V}]$")
# plt.ylabel(r"$I\,[\mathrm{mA}]$")

# fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
# fig.set_size_inches(6, 9)

# plt.grid(which = 'both')
# plt.savefig('../img/kennlinie1.pdf', bbox_inches='tight')
# plt.clf()

# plt.plot(U[0], I[0], "rx")
# plt.plot(U[0], .004+ 0 * U[0], "r-")
# plt.plot(U[1], I[1], "bx")
# plt.plot(U[1], .04+ 0 * U[1], "b-")

# plt.xlabel(r"$U\,[\mathrm{V}]$")
# plt.ylabel(r"$I\,[\mathrm{mA}]$")

# fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
# fig.set_size_inches(6, 9)

# plt.grid(which = 'both')
# plt.savefig('../img/kennlinie2.pdf', bbox_inches='tight')
# plt.clf()

U, I = U[4], I[4]
U = U[1:]
I = I[1:]
U += 1e-10
I += 1e-10
langmuir = lambda x, C, a: C + x * a

vor_werte = np.array([1, 1])
p, cov = curve_fit(langmuir, np.log(U), np.log(I), p0 = vor_werte, maxfev = 1000)

print("a = " + str(p[1]))
print("a_err = " + str(np.e ** p[1] * np.sqrt(cov[1][1])))

plt.plot(np.log(U), np.log(I), "kx")
plt.plot(np.log(U), langmuir(np.log(U), p[0], p[1]), "r-")
plt.legend(("Messwerte", "Regression"), "upper left")

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 5)

plt.grid(which = 'both')
plt.savefig('../img/langmuir.pdf', bbox_inches='tight')
plt.clf()

langmuir_nichtlinear = lambda x, C, a: C * (x ** a)

vor_werte = np.array([1, 1])
p, cov = curve_fit(langmuir_nichtlinear, U, I, p0 = vor_werte, maxfev = 1000)

print("a = " + str(p[1]))
print("a_err = " + str(np.sqrt(cov[1][1])))

plt.plot(U, I, "kx")
plt.plot(U, langmuir_nichtlinear(U, p[0], p[1]), "r-")
plt.legend(("Messwerte", "Regression"), "upper left")

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
fig.set_size_inches(9, 5)

plt.grid(which = 'both')
plt.savefig('../img/langmuir_nichtlinear.pdf', bbox_inches='tight')
plt.clf()