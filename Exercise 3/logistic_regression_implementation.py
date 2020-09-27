import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[1:100, [1, 3]]
y = iris.target[1:100]

class logistic_regression_implementation(object):
    def __init__(self, epocs = 100, eta = 0.1, random_state = 1):
        self.epocs = epocs
        self.eta = eta
        self.random_state = random_state

    def fit(self, x, y):
        np.random.RandomState(self.random_state)
        self.w = np.random.normal(loc = 0.0, scale = 1.0, size = x.shape[1] + 1)
        self.cost_ = []

        for i in range(self.epocs):
            net = np.dot(x, self.w[1:]) + self.w[0]
            output = 1. / (1. + np.exp(-net))
            error = (y - output)
            self.w[1:] += self.eta * np.dot(x.T, error)
            self.w[0] += self.eta * error.sum()
            cost = error.mean()
            cost = -np.dot(y, np.log(output)) - np.dot((1 - y), np.log(1 - output))
            self.cost_.append(cost)
        return (self)

    def predict(self, x):
        net = np.dot(x, self.w[1:]) + self.w[1]
        output = 1. / (1. + np.exp(-net))
        return (np.where(output >= 0.5, 1, 0))


model = logistic_regression_implementation(epocs = 20, eta = 0.05)
model.fit(x, y)
y_pred = model.predict(x)
error = np.sum(np.abs(y-y_pred))

# Decision line
# xx1 = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 0.1)
# xx2 = -(model.w[1] * xx1 + model.w[0]) / model.w[2]
#
# plt.plot(x[y_pred==0,0],x[y_pred==0,1],'g+',label='1')
# plt.plot(x[y_pred==1,0],x[y_pred==1,1],'bo',label='0')
# plt.legend(loc = 'upper left')
# plt.plot(xx1, xx2, 'm-')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.grid()
# plt.show()
#
# plt.plot(range(len(model.cost_)), model.cost_)
# plt.xlabel('Epochs')
# plt.ylabel('Error')
# plt.grid()
# plt.show()