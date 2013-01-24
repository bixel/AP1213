import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

g, e, B = np.genfromtxt("f_bekannt.txt", unpack = True)
b = e - g
f = 1 / (1 / g + 1 / b)
for i in f:
	print(str(round(i, 2)))

f_gesamt = f.sum() / f.size
print("f_quer = " + str(f_gesamt) + "mm")
f_fehler = np.sqrt(((f - f_gesamt) ** 2).sum() / (f.size * (f.size - 1)))
print("f_fehler = " + str(f_fehler) + "mm")

# V1_gesamt = (b / g).sum() / b.size
# print("V1_quer = " + str(V1_gesamt))
# V1_fehler = np.sqrt((((b / g) - V1_gesamt) ** 2).sum() / (b.size * (b.size - 1)))
# print("V1_fehler = " + str(V1_fehler))

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
print("Verh_mittel: " + str(round(verh_mittel, 3)))

# V2_gesamt = (B / G).sum() / B.size
# print("V2_quer = " + str(V2_gesamt))
# V2_fehler = np.sqrt((((B / G) - V2_gesamt) ** 2).sum() / (B.size * (B.size - 1)))
# print("V2_fehler = " + str(V2_fehler))

index = 0
while index < 10:
	x_theorie = np.arange(0, g[index], .1)
	plt.plot(x_theorie,  -b[index] * x_theorie / g[index] + b[index], "r-", linewidth = .1)
	index += 1
plt.plot(f_gesamt, f_gesamt, "k+", linewidth = 2)

fig = plt.gcf() #Gibt Referent auf die aktuelle Figur - "get current figure"
#fig.set_size_inches(9, 6)

plt.grid(which = 'both')
plt.savefig('../img/graph_bekannt.pdf', bbox_inches='tight')
plt.clf()