import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import random

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
x = mnist.data
y = mnist.target
y = y.astype(np.uint8())
print(x.shape)
print(y.shape)

# Split into training and testing
xtrain, xtest, ytrain, ytest = x[:60000], x[60000:], y[:60000], y[60000:]

fig, ax = plt.subplots(3, 5)
ax = ax.flatten()
ims = []
for i in range(len(ax)):
    ax[i].imshow(x[i].reshape(28,28), cmap='binary', interpolation='nearest', animated=True)
    ax[i].set_title(str(y[i]))
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
plt.show()

model = SGDClassifier(random_state=42, verbose=1, max_iter=10)
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)
print('Mean accuracy: ', model.score(xtest, ytest))
cm = confusion_matrix(ytest, y_pred)

cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])

plt.figure()
plt.matshow(cm, cmap=cmap)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i,j], va='center', ha='center')

plt.title('MNIST')
plt.xlabel('Predicted digits')
plt.ylabel('True digits')
plt.show()

print(model.coef_)
print(model.intercept_)