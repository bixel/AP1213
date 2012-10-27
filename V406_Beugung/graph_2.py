import numpy as np
import matplotlib.pyplot as plt

x, y = np.genfromtxt('daten_mr_2.txt', unpack=True)
# Datensatz in x und y einlesen

x -= 25
# x um 25 nach links verschieben (Maximum zentrieren)

x_ticks = range(-25, 26, 5)
# Schritte fuer x-Skalierung festlegen

plt.plot(x, y, 'kx')
plt.title('Messreihe 1')
plt.xlabel('x [mm]')
plt.ylabel('I [nA]')
plt.xticks(x_ticks)
plt.grid(b = True, which = "major")
plt.show()