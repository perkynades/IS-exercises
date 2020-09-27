import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# Upload iris dataset
iris = datasets.load_iris()

x = iris.data[:, [2, 3]]
y = iris.target
print(np.unique(y))
print(iris.target_names)
print(iris.feature_names)

# Standardize your feature vector
sc = StandardScaler()
sc.fit(x)
xstd = sc.transform(x)
print(x[1:2, :], xstd[1:2, :])

# Split dataset into training and testing sets
x = xstd
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 1, stratify = y)
print(np.bincount(ytrain), np.bincount(ytest))

# Select a model for training
model = Perceptron(max_iter = 20, eta0 = 0.002, random_state = 1)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
missclasified = (ytest != ypred).sum()
print('Misclassified samples = ', missclasified)
print('Model accuracy = ', (len(ytest) - missclasified) / len(ytest) * 100)
print('Model accuracy = ', model.score(xtest, ytest) * 100, '%')

print('Weights: \n', model.coef_)
print('Bias \n', model.intercept_)

# Plot decision regions in 2D space
x1 = np.arange(x[:, 0].min() - 1, x[:, 0].max() + 1, 0.1)
x2 = np.arange(x[:, 1].min() - 1, x[:, 1].max() + 1, 0.1)
x1g, x2g = np.meshgrid(x1, x2)
z = model.predict(np.array([x1g.flatten(), x2g.flatten()]).T)
plt.contourf(x1g, x2g, z.reshape(x1g.shape))
plt.scatter(x[y == 0, 0], x[y == 0, 1], color = 'red', marker = 'o')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color = 'm', marker = 'd')
plt.scatter(x[y == 2, 0], x[y == 2, 1], color = 'blue', marker = '+')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
