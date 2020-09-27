import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = MLPClassifier(activation='logistic',max_iter=100,hidden_layer_sizes=(2,),solver='lbfgs')
model.fit(x, y)

y_pred = model.predict(x)

print('coefs', model.coefs_)
print('predictions:', y_pred)

error = np.sum(y != y_pred).mean() * 100
print('Prediction error:', error)

# plot decision line
plt.figure()
x1 = np.arange(-0.1, 1.1, 0.01)
x1g, x2g=np.meshgrid(x1, x1)
yg=model.predict( np.array([x1g.flatten(), x2g.flatten()]).T)
plt.contourf(x1g, x2g, yg.reshape(x1g.shape))
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

# confusion matrix
cm =confusion_matrix(y, y_pred)
plt.figure()
plt.matshow(cm,cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
  for j in range(cm.shape[1]):
    plt.text(x=j, y=i, s=cm[i,j], va='center', ha='center')
plt.xlabel('Predicated label')
plt.ylabel('Acutal label')
plt.show()