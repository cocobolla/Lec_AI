import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_steps(X, y, iters=5000):
    def cost_func(X, y, theta):
        # For regression's hypothesis:  X.dot(theta)
        m = len(y)
        J = np.sum((X.dot(theta) - y)**2)/2/m
        return J

    m = len(y)
    theta = np.zeros(X.shape[1] + 1).reshape(-1, 1)  # Initialize Parameter with zero
    intercept_col = np.ones(m)
    # print(X)

    X_prime = np.zeros((X.shape[0], X.shape[1] + 1))
    X_prime[:, 1:] = X
    X_prime[:, 0] = intercept_col

    alpha = 0.01  # Learning rate is 0.01
    cost_history = [0] * iters  # Not use in Homework but useful

    for it in range(iters):
        hypothesis = X_prime.dot(theta)
        loss = hypothesis - y
        gradient = X_prime.T.dot(loss) / m
        theta = theta - alpha*gradient
        cost = cost_func(X_prime, y, theta)
        cost_history[it] = cost
    # plt.plot(range(iters), cost_history)
    # plt.show()
    return theta


np.random.seed(123)
Xtrain = 2 * np.random.rand(100, 3)
ytrain = 6 + Xtrain @ np.array([[3], [2], [5]]) + np.random.randn(100, 1)
Xtest = 2 * np.random.rand(20, 3)
ytest = 6 + Xtest @ np.array([[3], [2], [5]]) + np.random.randn(20, 1)

# Question.3 - (1): w_pred
w_pred = gradient_descent_steps(Xtrain, ytrain, iters=5000)
print("Fitted Parameter")
print(w_pred, end='\n\n')

# Question.3 - (2): MSE
m = len(ytest)
intercept_col = np.ones(m)
Xtest_prime = np.zeros((Xtest.shape[0], Xtest.shape[1] + 1))
Xtest_prime[:, 1:] = Xtest
Xtest_prime[:, 0] = intercept_col
a = Xtest_prime.dot(w_pred)

mse = np.sum((Xtest_prime.dot(w_pred) - ytest)**2) / m
print("MSE = {0:.4f}".format(mse))

