import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x_values, y_values, x, n):
    result = 0
    for i in range(n):
        numer = 1
        for j in range(n):
            if i != j:
                numer *= x - x_values[j]
        denom = 1
        for j in range(n):
            if i != j:
                denom *= x_values[i] - x_values[j]
        result += y_values[i] * numer / denom
    return result


x_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
y_values = np.array([7.0, 2.0, 0.0, 0.0, 0.0, 2.0, 7.0])

Y2 = []

for x in x_values:
    Y2.append(lagrange_interpolation(x_values, y_values, 0.3, 7))

txt1 = "The width of the plot changes with different standard deviations"
plt.title(
    "Applying Lagrange's polynomial to create a model that passes through all the data points"
)
plt.scatter(x_values, y_values, marker="x", c="blue", label="Original data points")
plt.plot(x_values, Y2, color="green", label="Lagrange's polynomial line")
plt.figtext(0.5, 0.01, txt1, wrap=True, horizontalalignment="center", fontsize=8)
plt.legend()
plt.show()
