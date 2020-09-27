import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = load_iris()
x = iris.data
y = iris.target
print(iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1, test_size=0.1)
np.bincount(y), np.bincount(y_train), np.bincount(y_test)

mlp = MLPClassifier(solver='sgd', random_state=0, hidden_layer_sizes=[5], alpha=0.5, max_iter=1000)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
print(np.sum(y_test != y_pred) / len(y_test)*100)

# Confusion matrix
def cm(y_test, y_pred, title):
    #cm = confusion_matrix(x_test, y_pred)
    cmap = ListedColormap(['lightgrey', 'silver', 'ghostwhite', 'lavender', 'wheat'])
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.matshow(cm, cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


cm(y_test, mlp.predict(x_test), title='Test')
cm(y_train, mlp.predict(x_train), title='Train')

