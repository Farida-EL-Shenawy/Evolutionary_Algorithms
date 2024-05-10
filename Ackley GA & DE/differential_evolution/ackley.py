import numpy as np
from matplotlib import pyplot as plt
import math
import tensorflow as tf

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


dimension = 3
popsize = 32
bound_lower = -33
bound_upper = 33

x = np.linspace(bound_lower, bound_upper, 100)
y = np.linspace(bound_lower, bound_upper, 100)
X, Y = np.meshgrid(x, y)

Z = Ackley([X, Y])
plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.contourf(X, Y, Z, popsize, cmap='viridis', alpha=0.8)
plt.axis('square')
plt.scatter([], [], s=0)  # Add an empty scatter plot with size 0 to avoid the error
plt.show()


