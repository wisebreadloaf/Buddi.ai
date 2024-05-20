# Importing the random
import random

# Importing the numerical Python library
import numpy as np

# Importing matplotlib for plotting the graph
import matplotlib.pyplot as plt


# mean squared error calculation for comparing the actual y with the prediction
def meanSquaredError(y: float, y_pred: float) -> float:
    # summation of squares of all the y with y predictions divided by the length of y using the np.mean method
    # return np.mean((y - y_pred) ** 2)
    return np.mean((y - y_pred) ** 2)


def betaCalculation(X: list[float], Y: list[float], n: int) -> int:
    # numpy array of X power 1 to 5
    Xtrans = [np.power(X, i) for i in range(n + 1)]

    # actual X values in form of X tranpose's transpose
    Xnew = np.transpose(Xtrans)

    # dot product of X transpose with X
    XTX = np.matmul(Xtrans, Xnew)

    # inverse of dot product of X Transpose with X
    XTXm1 = np.linalg.inv(XTX)

    # dot product of inverse of dot product of X transpose and X with X tranpose
    XTXinvintoXT = np.matmul(XTXm1, Xtrans)

    # Dot product of dot product of inverse of dot product of X transpose and X with X tranpose with Y
    Beta = np.matmul(XTXinvintoXT, Y)

    # array of beta elements
    return Beta


def I(Y_init):
    t = []
    for y in Y_init:
        if y == "B":
            t.append(1)
        else:
            t.append(0)
    return t


def D(Y_init):
    t = []
    for y in Y_init:
        if y > 0.5:
            t.append("B")
        else:
            t.append("G")
    return t


def S(Y_pred):
    return 1 / (1 + np.exp(-Y_pred))


# gets executed when the file is executed
# generating y values with the
# Generating 100 evenly spaced numbers between -5 and 5
# contains the whole population of data
X_init = np.array([1, 2, 3, 4, 5, -3, -4, -5, -6])
# the population is generated using the populationGenerator function for the whole input feature array X and returns an array of the population
Y_init = np.array(["B", "B", "B", "B", "B", "G", "G", "G", "G"])
Y_init = I(Y_init)
Y_mean = np.mean(Y_init)
Y_init = np.array(Y_init)
B0, B1 = betaCalculation(X_init, Y_init, 1)
Y_pred = B0 + B1 * X_init
Y_pred_dis = S(Y_pred)

x = (Y_mean - B0) / B1

slope_regression = B1

slope_perpendicular = -1 / slope_regression

x_mean = (Y_mean - B0) / B1
X_sig = S(X_init)
Y_sig = S(Y_pred)


def perpendicular_line(x):
    return slope_perpendicular * (x - x_mean) + Y_mean


x_range = np.linspace(min(X_init), max(X_init), 100)

plt.scatter(x, Y_mean, marker="x", c="g")
plt.scatter(X_init[:5], Y_init[:5], c="b", label="Data point with Label B")
plt.scatter(X_init[5:], Y_init[5:], c="g", label="Data point with Label G")
plt.plot(X_sig, Y_sig, c="black")
plt.plot(X_init, Y_pred_dis, marker=".", label="model fit line")
plt.plot(x_range, perpendicular_line(x_range), "g--", label="Perpendicular Line")
plt.plot(X_init, Y_pred)
plt.ylim(-6, 6)
plt.legend()
plt.show()
