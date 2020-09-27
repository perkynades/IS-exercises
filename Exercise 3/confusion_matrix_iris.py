import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from logistic_regression_implementation import logistic_regression_implementation

iris = load_iris()
x = iris.data[1:100, [1, 3]]
y = iris.target[1:100]

model = logistic_regression_implementation(epocs = 20, eta = 0.05)
model.fit(x, y)
y_pred = model.predict(x)
error = np.sum(np.abs(y-y_pred))

cmap = ListedColormap(['w', 'y', 'm'])

# Compute confusion matrix
y_true = y
class_names = iris.target_names
print(class_names[0: -1])

cm = confusion_matrix(y, y_pred)
print(cm)

plt.figure()
plt.matshow(cm, cmap = cmap)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x = j, y = i, s = cm[i, j], va = 'center', ha = 'center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()