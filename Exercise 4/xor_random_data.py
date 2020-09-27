import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Number of samples of each class
k = 100

# Define 4 clusters of input data
q = 0.6 # Offset of classes
x = np.random.rand(4*k, 2)
y = np.zeros((4*k, 1))

x[0:100, 0], x[0:100, 1], y[0:100] = x[0:100, 0] + q, x[0:100, 1] + q, 0
x[100:200, 0], x[100:200, 1], y[100:200] = x[100:200, 0] - q, x[100:200, 1] + q, 1
x[200:300, 0], x[200:300, 1], y[200:300] = x[200:300, 0] - q, x[200:300, 1] - q, 0
x[300:400, 0], x[300:400, 1], y[300:400] = x[300:400, 0] + q, x[300:400, 1] - q, 1

""" plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show() """

# MLPClassifier to classify the data into two classes
model = MLPClassifier(activation='logistic', max_iter=10000, hidden_layer_sizes=(20,), solver='lbfgs')
model.fit(x, y)

y_pred = model.predict(x)

print('Coefs:', model.coefs_)

error = np.sum(y != y_pred).mean() * 100
print('Prediction error;', error)

# Plot decision line
#plt.figure()
x1 = np.arange(x[:,0].min(), x[:,1].max(), 0.01)
x1g, x2g = np.meshgrid(x1, x1)
yg = model.predict(np.array([x1g.flatten(), x2g.flatten()]).T)
#plt.contourf(x1g, x2g, yg.reshape(x1g.shape))
#plt.scatter(x[:,0], x[:,1], c=y)
#plt.show()

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure()
plt.matshow(cm, cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i,j], va='center', ha='center')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()