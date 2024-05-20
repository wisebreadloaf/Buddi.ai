# Importing the numerical Python library
import numpy as np

# Importing matplotlib for plotting the graph
import matplotlib.pyplot as plt


def lagrange_interpolation(x_values, y_values, x):
    """
    Function for Lagrange interpolation.
    This function takes in three arguments:
    x_values: a list or array of x values
    y_values: a list or array of y values corresponding to the x values
    x: the x value at which the interpolated y value is to be calculated
    """
    n = len(y_values)
    result = 0

    for i in range(n):
        numer = 1
        denom = 1
        for j in range(n):
            if i != j:
                numer *= x - x_values[j]
                denom *= x_values[i] - x_values[j]
        result += y_values[i] * numer / denom
    return result


# Sample data points
x_values = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
y_values = np.array([7.0, 2.0, 0.0, 0.0, 0.0, 2.0, 7.0])

# Calculating the y values for the given x values using Lagrange interpolation
Y2 = []
for x in x_values:
    Y2.append(lagrange_interpolation(x_values, y_values, x))

# Text to be displayed on the plot
txt1 = "The width of the plot changes with different standard deviations"

# Plotting the original data points and the Lagrange interpolation curve
plt.title(
    "Applying Lagrange's polynomial to create a model that passes through all the data points"
)
plt.scatter(x_values, y_values, marker="x", c="red", label="Original data points")
plt.plot(x_values, Y2, color="blue", label="Lagrange's polynomial line")
plt.figtext(0.5, 0.01, txt1, wrap=True, horizontalalignment="center", fontsize=8)
plt.legend()
plt.show()
