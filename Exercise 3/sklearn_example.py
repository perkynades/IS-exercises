import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
x = iris.data[0:100, [2, 3]]
y = iris.target[0:100]

print(x[0:5, :])
print('Class labels', np.unique(y))
labels = ['setosa', 'versicolor', 'virginica']

# Standarization
sc = StandardScaler()
sc.fit(x)
xstd = sc.transform(x)
print(xstd[0:5, :])

# Import a model
model = Perceptron(max_iter = 100, eta0 = 0.01, random_state = 1)
model.fit(xstd, y)
y_pred = model.predict(x)

print(model.coef_, model.intercept_)

error = np.mean((y-y_pred)**2)
print(error)

# Plot decision line
# plt.scatter(xstd[y == 0, 0], xstd[y == 0, 1], color = 'red', marker = 'o', label = labels[0])
# plt.scatter(xstd[y == 1, 0], xstd[y == 1, 1], color = 'blue', marker = '+', label = labels[1])
#
# xx = np.arange(xstd[:, 0].min(), xstd[:, 0].max(), 0.1)
# yy = -model.coef_[0, 0] / model.coef_[0, 1] * xx - model.intercept_ / model.coef_[0, 1]
# plt.plot(xx, yy, 'g-')
# plt.grid()
# plt.show()

def plot_decision_region(x, y):
    x1 = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 0.1)
    x2 = np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, 0.1)

    xg1, xg2 = np.meshgrid(x1, x2)

    z = model.predict(np.array([xg1.ravel(), xg2.ravel()]).T)

    plt.contourf(xg1, xg2, z.reshape(xg1.shape))

    plt.scatter(x[y == 0, 0], x[y == 0, 1], color = 'yellow', marker = 'o')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color = 'black', marker = 'd')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

plot_decision_region(xstd, y_pred)