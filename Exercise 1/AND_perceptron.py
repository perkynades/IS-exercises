import numpy as numpy

def unit_step(v):
    if v >= 0:
        return 1
    else:
        return 0

def perceptron(x, w, b):
    v = numpy.dot(w, x) + b
    y = unit_step(v)
    return y

def AND_perceptron(x):
    w = numpy.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)

AND_example1 = numpy.array([1, 1])
AND_example2 = numpy.array([1, 0])
AND_example3 = numpy.array([0, 1])
AND_example4 = numpy.array([0, 0])

print("AND({}, {}) = {}".format(1, 1, AND_perceptron(AND_example1)))
print("AND({}, {}) = {}".format(1, 0, AND_perceptron(AND_example2)))
print("AND({}, {}) = {}".format(0, 1, AND_perceptron(AND_example3)))
print("AND({}, {}) = {}".format(0, 0, AND_perceptron(AND_example4)))

