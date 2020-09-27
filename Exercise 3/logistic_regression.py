import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x = np.array([29, 15, 33, 28, 39, 44, 31, 19, 9, 24, 32, 31, 37, 35])
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

log_reg = LogisticRegression()

log_reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))

print(log_reg.coef_)
print(log_reg.intercept_)

xx = np.arange(5, 50, 0.1).reshape(-1, 1)
y_pred = log_reg.predict(xx)

def sig(x):
    return (np.exp(x) / (1 + np.exp(x)))

net = log_reg.intercept_ + log_reg.coef_ * xx

plt.plot(x, y, 'bd')
plt.plot(xx, sig(net), 'g-')
plt.plot(xx, y_pred, 'r-')
plt.xlabel('Hours studied')
plt.ylabel('PASS / FAIL')
plt.grid()
plt.show()