import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
import math

x = numpy.arange(-1, 1, 0.001)

def tanH(val):
    return (math.e**val - math.e**(-val)) / (math.e**val + math.e**(-val))

ytanh = numpy.zeros(x.size)

for i in range(len(x)):
    ytanh[i] = tanH(x[i])

y_p = numpy.diff(ytanh) / numpy.diff(x)
x_p = (numpy.array(x)[:-1] + numpy.array(x)[1:]) / 2
pyplot.plot(x, ytanh, label="tanh")
pyplot.plot(x_p, y_p, c='r', label="derivative")
pyplot.legend()
pyplot.show()