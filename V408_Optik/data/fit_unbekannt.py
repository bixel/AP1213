import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

g, b, B = np.genfromtxt("f_unbekannt.txt", unpack = True)
e = b + g
f = 1 / (1 / g + 1 / b)
for i in f:
	print(str(round(i, 2)))

f_gesamt = f.sum() / f.size
print("f_quer = " + str(f_gesamt) + "mm")
f_fehler = np.sqrt(((f - f_gesamt) ** 2).sum() / (f.size * (f.size - 1)))
print("f_fehler = " + str(f_fehler) + "mm")

print("Abbildungsmassstaebe b/g:")
for i in range(0, b.size):
	print(str(round(b[i] / g[i] ,3)))

G = 27.5
print("Abbildungsmassstaebe B/G:")
for i in range(0, B.size):
	print(str(round(B[i] / G ,3)))

verh_mittel = 0
print("Verhaetlnis b/g / B/G:")
for i in range(0, B.size):
	verh = (b[i] / g[i]) / (B[i] / G)
	verh_mittel += verh
	print(str(round(verh, 3)))
verh_mittel /= i + 1
print("Verh_mittel: " + str(round(verh_mittel, 4)))

index = 0
while index < 10:
	x_theorie = np.arange(0, g[index], .1)
	plt.plot(x_theorie,  -b[index] * x_theorie / g[index] + b[index], "r-", linewidth = .1)
	index += 1
plt.plot(f_gesamt, f_gesamt, "k+", linewidth = 2)

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
#fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/graph_unbekannt.pdf', bbox_inches='tight')
plt.clf()