import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression

# Generate data
x, y = make_circles(n_samples=500, noise=0.02)
print(x.shape)

# Visualise data
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x[y==1, 0], x[y==1, 1], marker = '.', color='red')
plt.scatter(x[y==0, 0], x[y==0, 1], marker = '.', color='blue')
ax.set_aspect('equal')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()

# Add a new dimension to X
x1 = x[:, 0].reshape((-1, 1))
x2 = x[:, 1].reshape((-1, 1))
x3 = (x1**2 + x2**2) # Introducing a higher order feature

X = np.hstack((x1, x3))
print(X.shape)

#Visualise data for higher order dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c=y, depthshade=True)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.show()

# Fit a logistic regression (classification) model
model = LogisticRegression()

X = np.hstack((x1**2, x2**2))

model.fit(X, y)

y_pred = model.predict(X)
print(model.coef_)
print(model.intercept_)

error = np.mean(y-y_pred)
print(error)

xd1 = np.arange(x1.min(), x1.max(), 0.001)
xd2 = -(model.coef_[0,0] * xd1**2 + model.intercept_) / model.coef_[0,1]
xd2 = np.sqrt(xd2)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x1, x2, c=y)
plt.plot(xd1, xd2, 'k-', xd1, -xd2, 'k-')
ax.set_aspect('equal')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.show()