import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot

x = numpy.arange(-1, 1, 0.001)

def linear(x, a = 1, b = 0):
    return a * x + 1

ylin = numpy.zeros(x.size)

for i in range(len(x)):
    ylin[i] = linear(x[i])

y_p = numpy.diff(ylin) / numpy.diff(x)
x_p = (numpy.array(x)[:-1] + numpy.array(x)[1:]) / 2

pyplot.plot(x, ylin)
pyplot.plot(x_p, y_p, c='r')
pyplot.show()