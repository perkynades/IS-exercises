import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)

x = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, -1)

xn = (x - x.mean(axis = 0)) / x.std(axis = 0)

class Adaline(object):

    def __init__(self, epochs = 100, eta = 0.1):
        self.epochs = epochs
        self.eta = eta

    def train(self, training_inputs, training_labels, mode = 'SGD'):
        x = training_inputs
        t = training_labels
        self.cost = []
        self.w = random.rand(training_inputs.ndim + 1)

        for i in range(self.epochs):
            if mode == 'BGD':
                net = np.dot(x, self.w[1:]) + self.w[0]
                y = net
                error = (t - y)
                self.w[1:] += self.eta * (np.dot(error, x)).mean()
                cost = 0.5 * (error**2).sum()
                self.cost.append(cost)
            elif mode == 'SGD':
                cost = 0
                for j in range(len(x)):
                    net = np.dot(x[j, :], self.w[1:]) + self.w[0]
                    y = net
                    error = (t[j] - y)
                    self.w[1:] += self.eta * np.dot(error, x[j, :])
                    self.w[0] += self.eta * error
                    cost += error**2
                self.cost.append(cost / len(x))
            elif mode == 'MGD':
                net = np.dot()
        return self

    def predict(self, inputs):
        net = np.dot(inputs, self.w[1:]) + self.w[0]
        return np.where(net >= 0, 1, -1)

model = Adaline(epochs = 10, eta = 0.1)
model.train(xn, y, mode = 'SGD')
predictions = model.predict(xn)

printout = 0
if printout:
    print(model.cost)
    print(model.w)
    print(predictions)

plt.subplot(211)
plt.plot(range(1, len(model.cost) + 1), model.cost, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training Error')
plt.grid()

plt.subplot(212)
plt.scatter(xn[predictions == 1, 0], xn[predictions == 1, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(xn[predictions == -1, 0], xn[predictions == -1, 1], color = 'blue', marker = 'x', label = 'versicolor')
xx1 = np.arange(xn[:, 0].min()-2, xn[:, 0].max() + 2, 0.1)
xx2 = -model.w[1] / model.w[2] * xx1 - model.w[0] / model.w[2]
plt.plot(xx1, xx2, 'g--')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc = 'upper left')
plt.grid()

plt.show()
