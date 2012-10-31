from pylab import *
import matplotlib.pyplot as plt

x = [-.1, -.6, -1.1, -1.6, -2.1, -2.6, -3.1, -3.6, -4.1, -4.6]
y = [-4, 6, 16, 32, 54, 76, 120, 176, 280, 396]



i = 0
for var in y:
	x[i] += 4.6
	y[i] = log(var + 17)
	i += 1

(koeffs, errors, bla, blubb, kaka) = polyfit(x, y, 1, full=True)

print(koeffs)
print(errors)

#print('b = ' + str(b))
#print('10^log(b) = ' + str(10**log(b)))
#print('m = ' + str(m))
#print('1/m = ' + str(1/m))

#a = arange(0, 5, 0.5)

plt.plot(x, y, 'x')
#plt.plot(a, m*a + b, '-')
plt.ylabel('U_C')
plt.xlabel('t')
plt.grid(axis='both')



plt.show()