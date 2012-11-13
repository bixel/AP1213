import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

#messungen = ('29_fein_1', '29_fein_2', '29_grob', '32_fein', '32_grob', '41_fein', '41_grob', '79_fein', '79_grob', '80_fein', '80_grob')
messungen = ('32_fein', '32_grob')

for messung in messungen:
	print(messung)
	x, y = np.genfromtxt('../data/' + messung + '.txt', unpack = True)
	#x /= 2
	print(max(y))
	print('')

	x = 3084.183725 / np.sin(x * (2 * np.pi / 360))

	plt.plot(x, y, 'r-')
	plt.xlabel('Energie [eV]')
	plt.ylabel('Intensitaet [1/s]')

	plt.grid(b = True, which = "major")

	plt.savefig('graph_' + messung + '.png')
	plt.clf() #clearfigure - damit wird der Graph zurueckgesetzt fuers naechste speichern