import numpy as np


def softmax(theta, data):
    result = []
    for i in range(np.shape(data)[0]):
        x = theta @ data[i, :-1]
        p = np.exp(x) / np.exp(x).sum()
        result.append(p)
    final_index = np.argmax(result, axis=1)
    return final_index, result


def cost_function(theta, data):
    y = data[:, -1]
    result = softmax(theta, data)[1]
    cost_x = []
    for i in range(np.shape(data)[0]):
        cost_x.append(np.log(result[i][int(y[i] - 1)]))
    cost_x = np.array(cost_x)
    cost = -(np.sum(cost_x)) / np.shape(cost_x)[0]
    return cost


def main():
    np.random.seed(123)
    tradint = np.hstack([np.ones((5, 1)),
                         np.around(np.random.randn(5, 4), 3),
                         np.random.randint(1, 4, (5, 1))])

    theta1 = np.array([[5, 2, 3, 1, 4],
                       [2, 4, 3, 1, 2],
                       [3, 4, 1, 5, 4]])

    soft_theta1 = softmax(theta1, tradint)
    print("theta2: {}".format(soft_theta1[0] + 1))

    theta2 = np.array([[5.5, 2, 3, 1.5, 4],
                       [2, 3.5, 2.5, 1, 1.5],
                       [3, 4, 1, 5, 4]])
    soft_theta2 = softmax(theta2, tradint)

    print("theta2: {}".format(soft_theta2[0] + 1))

    cost1 = cost_function(theta1, tradint)
    print("theta1: {}".format(cost1))

    t1_r = (softmax(theta1, tradint))
    t2_r = softmax(theta2, tradint)
    t1_c = cost_function(theta1, tradint)
    t2_c = cost_function(theta2, tradint)
    print(t1_r)
    print(t2_r)
    print(t1_c)
    print(t2_c)


if __name__ == '__main__':
    main()
