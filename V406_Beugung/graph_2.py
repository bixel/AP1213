import numpy as np
import matplotlib.pyplot as plt

x, y = np.genfromtxt('daten_mr_2.txt', unpack=True)
# Datensatz in x und y einlesen

x -= 25
# x um 25 nach links verschieben (Maximum zentrieren)

y /= max(y)
# y normieren

x_ticks = np.arange(-25, 26, 5)
y_ticks = np.arange(0, 1.2, .1)
# Schritte fuer Skalierung festlegen

plt.plot(x, y, 'kx')
plt.title('Messreihe 2')
plt.xlabel('x [mm]')
plt.ylabel('I [nA]')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(b = True, which = "major")
plt.show()