import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit

messungen = ('29_fein_1', '29_fein_2', '29_grob', '32_fein', '32_grob', '41_fein', '41_grob', '79_fein', '79_grob', '80_fein', '80_grob', 'adjust')
#messungen = ('79_fein', '79_grob')

for messung in messungen:
	print messung
	x, y = np.genfromtxt('../data/' + messung + '.txt', unpack = True)

	x = 3084.183725 / np.sin( 2 * x * (np.pi / 360))

	plt.plot(x, y, 'r-')
	plt.xlabel('Energie')
	plt.ylabel('Intensitaet')

	plt.savefig('graph_' + messung + '.png')
	plt.clf() #clearfigure - damit wird der Graph zurueckgesetzt fuers naechste speichern