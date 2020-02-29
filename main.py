import numpy as np
import matplotlib.pyplot as plt
import cvxopt

N = 100

positive = []
negative = []

mean1 = [-2, 2]
mean2 = [2, -2]
cov = [[1.0, 0.3], [0.3, 1.0]]

positive.extend(np.random.multivariate_normal(mean1, cov, N//2))
negative.extend(np.random.multivariate_normal(mean2, cov, N//2))
X = np.vstack((positive, negative))
    
y = []
for i in range(N//2):
    y.append(1.0)
for i in range(N//2):
    y.append(-1.0)
y = np.array(y)

_Q = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        _Q[i, j] = y[i]*y[j]*np.dot(X[i], X[j])

Q = cvxopt.matrix(_Q)
p = cvxopt.matrix(-np.ones(N))
G = cvxopt.matrix(-np.eye(N))
h = cvxopt.matrix(np.zeros(N))
A = cvxopt.matrix(y.reshape((1, N)))
b = cvxopt.matrix(0.0)

solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
alpha = np.array(solution['x']).flatten()

top2_sv_indices = alpha.argsort()[-2:]
sv_indices = alpha > 1e-6
W = np.dot(alpha[sv_indices] * y.flatten()[sv_indices], X[sv_indices])
bias = -np.dot(X[top2_sv_indices], W).mean()

xs = np.array([X.min(), X.max()])
ys = np.array([(-W[0]*xs[0]-bias)/W[1], (-W[0]*xs[1]-bias)/W[1]])

plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], label="positive")
plt.scatter(X[:, 0][y == -1], X[:, 1][y == -1], label="negative")
for coordinate in X[sv_indices]:
    plt.annotate('sv', coordinate)
plt.plot(xs, ys, color = "sienna")
plt.legend()
plt.grid()
plt.show()
