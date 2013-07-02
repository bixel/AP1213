import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import poisson
import datetime as dt
import pylab
from uncertainties import ufloat
from uncertainties import unumpy

g = 9.81

gleichungEinseitig = lambda x, A: A * (L * x**2 - x**3 / 3)
gleichungBeidseiig = lambda x, A: A * (3 * L**2 * x - 4 * x**3)
#lineareGleichung = lambda x, a: a * x
lineareGleichungOffset = lambda x, a, b: a * x + b

#############
# MESSUNG 1 #
#############
m = 542.5
Ms = 167.1
L = 530

x, D0, Dm = np.genfromtxt("messung1.txt", unpack = True)
D = D0 - Dm
D = D[::-1]
x *= 10
x = x[::-1]

xTheorie = np.arange(0, 530, (np.max(x) - np.min(x)) / 1000)

a, b = np.genfromtxt("ausmasse1.txt", unpack = True)
c = np.append(a, b)
cMittel = np.sum(c) / np.size(c)
cMittelError = np.sqrt(1. / (np.size(c) * (np.size(c) - 1.)) * np.sum((c - cMittel) ** 2))

koeffizienten, varianzen = curve_fit(lineareGleichungOffset, (L * x**2 - x**3 / 3), D)

A = koeffizienten[0]
AError = np.sqrt(varianzen[0][0])

F = m * g / 1000
I = cMittel**4 / 12.
IError = cMittel**3 / 3 * cMittelError

E = F / (2. * A * I)
EError = F / (2 * A * I) * np.sqrt((IError / I)**2 + (AError / A)**2)

print("Messung 1 \n####################################################")
print("\tm = " + str(m))
print("\tMs = " + str(Ms))
print("\tL = " + str(L) + "\n")

print("\tx [mm]\t\tD0 [mm]\t\tDm [mm]\t\tD [mm]:")
for i in range(0, np.size(x), 1):
	print("\t" + str(x[i]) + "\t\t" + str(D0[i]) + "\t\t" + str(Dm[i]) + "\t\t" + str(D[i]))
print("\n")

print("\tcMittel = " + str(round(cMittel, 3)) + "+-" + str(round(cMittelError, 3)))

print("\tF = " + str(round(F, 3)))
print("\tI = " + str(round(I, 3)) + "+-" + str(round(IError, 3)))

print("\tA = " + str(A) + "+-" + str(AError))
print("\tE = " + str(E) + "+-" + str(EError))

fig = plt.gcf()

plt.plot((L * x**2 - x**3 / 3), D, "kx")
plt.plot((L * xTheorie**2 - xTheorie**3 / 3), lineareGleichungOffset((L * xTheorie**2 - xTheorie**3 / 3), koeffizienten[0], koeffizienten[1]), "r-")

plt.ylabel(r"$D\,[\mathrm{mm}]$")
plt.xlabel(r"$Lx^2 - \frac{x^3}{3}\,\left[\mathrm{mm}^3\right]$")
plt.legend(["lin. Messwerte", "Ausgleichsgerade"], "upper left")

plt.xlim([0, (L * 530**2 - 530**3 / 3)])
plt.ylim([0, 5.0])

plt.grid(which = "both")
plt.savefig("../img/plot1.pdf", bbox_inches = "tight")
fig.set_size_inches(10 * .7, 6 * .7)

plt.clf()



#############
# MESSUNG 2 #
#############
m = 267.8
Ms = 394.5
L = 530

x, D0, Dm = np.genfromtxt("messung2.txt", unpack = True)
D = D0 - Dm
D = D[::-1]
x *= 10
x = x[::-1]

r = np.genfromtxt("ausmasse2.txt", unpack = True)
r /= 2
rMittel = np.sum(r) / np.size(r)
rMittelError = np.sqrt(1. / (np.size(r) * (np.size(r) - 1.)) * np.sum((r - rMittel) ** 2))

koeffizienten, varianzen = curve_fit(lineareGleichungOffset, (L * x**2 - x**3 / 3), D)

A = koeffizienten[0]
AError = np.sqrt(varianzen[0][0])

F = m * g / 1000
I = np.pi / 4 * rMittel ** 4
IError = cMittel**3 * np.pi * rMittelError

E = F / (2. * A * I)
EError = F / (2 * A * I) * np.sqrt((IError / I)**2 + (AError / A)**2)

print("\nMessung 2 \n####################################################")
print("\tm = " + str(m))
print("\tMs = " + str(Ms))
print("\tL = " + str(L) + "\n")

print("\tx [mm]\t\tD0 [mm]\t\tDm [mm]\t\tD [mm]:")
for i in range(0, np.size(x), 1):
	print("\t" + str(x[i]) + "\t\t" + str(D0[i]) + "\t\t" + str(Dm[i]) + "\t\t" + str(D[i]))
print("\n")

print("\trMittel = " + str(round(rMittel, 3)) + "+-" + str(round(rMittelError, 3)))
print("\tF = " + str(round(F, 3)))
print("\tI = " + str(round(I, 3)) + "+-" + str(round(IError, 3)))

print("\tA = " + str(A) + "+-" + str(AError))
print("\tE = " + str(E) + "+-" + str(EError))

fig = plt.gcf()

plt.plot((L * x**2 - x**3 / 3), D, "kx")
plt.plot((L * xTheorie**2 - xTheorie**3 / 3), lineareGleichungOffset((L * xTheorie**2 - xTheorie**3 / 3), koeffizienten[0], koeffizienten[1]), "r-")

plt.ylabel(r"$D\,[\mathrm{mm}]$")
plt.xlabel(r"$Lx^2 - \frac{x^3}{3}\,\left[\mathrm{mm}^3\right]$")
plt.legend(["lin. Messwerte", "Ausgleichsgerade"], "upper left")

plt.xlim([0, (L * 530**2 - 530**3 / 3)])
plt.ylim([0, 3.5])

plt.grid(which = "both")
fig.set_size_inches(10 * .7, 6 * .7)
plt.savefig("../img/plot2.pdf", bbox_inches = "tight")

plt.clf()


#############
# MESSUNG 3 #
#############
m = 1648.3
Ms = 394.5
L = 555

x, D0, Dm = np.genfromtxt("messung3.txt", unpack = True)
D = D0 - Dm
x *= 10

DLinks = D[0:13:1]
DRechts = D[13:26:1]
xLinks = x[0:13:1]
xRechts = x[13:26:1]
xTheorieLinks = np.arange(0, 277.5, .1)
xTheorieRechts = np.arange(277.5, 555, .1)

koeffizienten1, varianzen1 = curve_fit(lineareGleichungOffset, (3 * L**2 * xLinks - 4 * xLinks**3), DLinks)

ALinks = koeffizienten1[0]
ALinksError = np.sqrt(varianzen1[0][0])

F = m * g / 1000
I = np.pi / 4 * rMittel ** 4
IError = cMittel**3 * np.pi * rMittelError

ELinks = F / (48. * ALinks * I)
ELinksError = F / (48 * ALinks * I) * np.sqrt((IError / I)**2 + (ALinksError / ALinks)**2)

#b = - F / (48 * E * I) * L**3
koeffizienten2, varianzen2 = curve_fit(lineareGleichungOffset, (4 * xRechts**3 - 12 * L * xRechts**2 + 9 * L**2 * xRechts - L**3), DRechts)

ARechts = koeffizienten2[0]
ARechtsError = np.sqrt(varianzen2[0][0])

ERechts = F / (48. * ARechts * I)
ERechtsError = F / (48 * ARechts * I) * np.sqrt((IError / I)**2 + (ARechtsError / ARechts)**2)

EMittel = (ERechts + ELinks) / 2
EMittelError = np.sqrt(ELinksError**2 + ERechtsError**2)

print("\nMessung 3 \n####################################################")
print("\tm = " + str(m))
print("\tMs = " + str(Ms))
print("\tL = " + str(L) + "\n")

print("\tx [mm]\t\tD [mm]:")
for i in range(0, np.size(x), 1):
	print("\t" + str(x[i]) + "\t\t" + str(D[i]))
print("\n")

print("\trMittel = " + str(round(rMittel, 3)) + "+-" + str(round(rMittelError, 3)))
print("\tF = " + str(round(F, 3)))

#print("\tA = " + str(A) + "+-" + str(AError))
print("\tELinks = " + str(ELinks) + "+-" + str(ELinksError))
print("\tERechts = " + str(ERechts) + "+-" + str(ERechtsError))
print("\tEMittel = " + str(EMittel) + "+-" + str(EMittelError))



fig = plt.gcf()

plt.plot((3 * L**2 * xLinks - 4 * xLinks**3), DLinks, "rx")
plt.plot((3 * L**2 * xTheorieLinks - 4 * xTheorieLinks**3), lineareGleichungOffset((3 * L**2 * xTheorieLinks - 4 * xTheorieLinks**3), koeffizienten1[0], koeffizienten1[1]), "r-")

plt.plot((4 * xRechts**3 - 12 * L * xRechts**2 + 9 * L**2 * xRechts - L**3), DRechts, "bx")
plt.plot((4 * xTheorieRechts**3 - 12 * L * xTheorieRechts**2 + 9 * L**2 * xTheorieRechts - L**3), lineareGleichungOffset((4 * xTheorieRechts**3 - 12 * L * xTheorieRechts**2 + 9 * L**2 * xTheorieRechts - L**3), koeffizienten2[0], koeffizienten2[1]), "b-")

plt.ylabel(r"$D\,[\mathrm{mm}]$")
plt.xlabel(r"$\chi\,,\,\phi\,\left[\mathrm{mm}^3\right]$")
plt.legend(["lin. Messwerte re.", "Ausgleichsgerade", "lin. Messwerte li.", "Ausgleichsgerade"], "upper left")

# plt.xlim([0, (L * 530**2 - 530**3 / 3)])
# plt.ylim([0, 3.5])

plt.grid(which = "both")
plt.savefig("../img/plot3.pdf", bbox_inches = "tight")

plt.clf()


