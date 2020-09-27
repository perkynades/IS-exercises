import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as py_plot
import io

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pandas.read_csv("datasets/Salary_Data.csv")

x = data["YearsExperience"]
y = data["Annual Salary(Naira)"]

def estimate_coef(x, y):
    n = numpy.size(x)

    m_x, m_y = numpy.mean(x), numpy.mean(y)

    SS_xy = numpy.sum(y*x) - n*m_y*m_x
    SS_xx = numpy.sum(x*x) - n*m_x*m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)

def plot_regression_line(x, y, b):
    py_plot.scatter(x, y, color = "m", marker = "o", s = 30)

    y_pred = b[0] + b[1]*x

    py_plot.plot(x, y_pred, color = "g")

    py_plot.xlabel("x")
    py_plot.ylabel("y")

    py_plot.show()

b = estimate_coef(x, y)
plot_regression_line(x, y, b)