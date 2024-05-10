import numpy as np


def Ackley(d):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = 0
    sum2 = 0
    for i in range(len(d)):
        sum1 += d[i] ** 2
        sum2 += np.cos(c * d[i])
    term1 = -a * np.exp(-b * np.sqrt(sum1 / len(d)))
    term2 = -np.exp(sum2 / len(d))

    return term1 + term2 + a + np.exp(1)