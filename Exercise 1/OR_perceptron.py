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

def OR_perceptron(x):
    w = numpy.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

OR_example1 = numpy.array([1, 1])
OR_example2 = numpy.array([1, 0])
OR_example3 = numpy.array([0, 1])
OR_example4 = numpy.array([0, 0])

print("AND({}, {}) = {}".format(1, 1, OR_perceptron(OR_example1)))
print("AND({}, {}) = {}".format(1, 0, OR_perceptron(OR_example2)))
print("AND({}, {}) = {}".format(0, 1, OR_perceptron(OR_example3)))
print("AND({}, {}) = {}".format(0, 0, OR_perceptron(OR_example4)))