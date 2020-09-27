import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot

x = numpy.arange(-1, 1, 0.001)

def hardlim(val):
    if val > 0:
        return 1
    else:
        return 0

y = numpy.zeros(x.size)

for i in range(len(x)):
    y[i] = hardlim(x[i])

data = {'x' : x, 'y' : y}
y_p = numpy.diff(data['y']) / numpy.diff(data['x'])
x_p = (numpy.array(data['x'])[:-1] + numpy.array(data['x'])[1:]) / 2

pyplot.ylim(-0.15, 1.15)
pyplot.plot(x, y, label='hardlim')
pyplot.plot(x_p, y_p, c = 'r', label = 'derivative', alpha= 0.4)
pyplot.legend()
pyplot.show()