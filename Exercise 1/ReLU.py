import numpy as numpy
import pandas as pandas
import matplotlib.pyplot as pyplot
import math

x = numpy.arange(-1, 1, 0.001)

def ReLU(x):
    return max(0, x)

yrelu = numpy.zeros(x.size)

for i in range(len(x)):
  yrelu[i] = ReLU(x[i])

y_p = numpy.diff(yrelu) / numpy.diff(x)
x_p = (numpy.array(x)[:-1] + numpy.array(x)[1:]) / 2

pyplot.plot(x, yrelu, label='ReLU')
pyplot.plot(x_p, y_p, c='r', label="derivative", alpha=.4)
pyplot.legend()
pyplot.show()