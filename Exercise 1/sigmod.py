import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
import math

x = numpy.arange(-1, 1, 0.001)

def sigmod(val):
    return 1 / (1+(math.e**(-val)))

ysig = numpy.zeros(x.size)

for i in range(len(x)):
    ysig[i] = sigmod(x[i])

y_p = numpy.diff(ysig) / numpy.diff(x)
x_p = (numpy.array(x)[:-1] + numpy.array(x)[1:]) / 2

pyplot.plot(x, ysig, label='sigmoid')
pyplot.plot(x_p, y_p, c='r', label='derivative')
pyplot.legend()
pyplot.show()