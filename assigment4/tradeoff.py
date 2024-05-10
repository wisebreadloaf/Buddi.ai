# Importing the numerical Python library
import numpy as np

# Importing matplotlib for plotting the graph
import matplotlib.pyplot as plt

# random module
import random


# Function to generate a dataset based on the input x
def datasetGenerator(x: int) -> int:
    return (
        (2 * (x**4))
        - (3 * (x**3))
        + (7 * (x**2))
        - (23 * (x))
        + 8
        # Adding noise to the data
        + np.random.normal(0, 100)
    )


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
    B_four = np.matmul(XTXinvintoXT, Y)

    # array of beta elements
    return B_four


# Function for Lagrange interpolation
def lagrangeInterpolation(x_values, y_values, x, n):
    result = 0
    # initialising a for loop for i, j in the lagrange equation
    for i in range(n):
        numer = 1
        denom = 1
        for j in range(n):
            # calculating the numerator and denominator seperately
            if i != j:
                numer *= x - x_values[j]
                denom *= x_values[i] - x_values[j]
        # adding the multiplication of yi into (numerator/ denominator) to result
        result += y_values[i] * numer / denom

    # return the result prediction of the lagrange model
    return result


# mean squared error calculation for comparing the actual y with the prediction
def MeanSquaredError(y: float, y_pred: float) -> float:
    # summation of squares of all the y with y predictions divided by the length of y
    return sum((y - y_pred) ** 2 / len(y))


def biasVarianceTradeoff(
    X_train: list[float],
    Y_train: list[float],
    X_test: list[float],
    Y_test: list[float],
    deg: int,
    betaArray: list[list[float]],
) -> tuple[list[float]]:
    # empty bias and varinace array initialised
    bias = []
    variance = []

    # model predictions for the training dataset is computed for all the models till the degree size of deg+1
    for i in range(deg + 1):
        # The predicted value of the training dataset is computed using the respective models
        y_pred_train = np.array(
            sum(
                betaArray[i][j] * np.power(X_train, j) for j in range(len(betaArray[i]))
            )
        )

        # The predicted value of the testing dataset is computed using the models computed using the training dataset
        y_pred_test = np.array(
            sum(betaArray[i][j] * np.power(X_test, j) for j in range(len(betaArray[i])))
        )

        # The mean squared error is computed for the training dataset
        yBias = MeanSquaredError(Y_train, y_pred_train)
        # The mean squared error is computed for the testing dataset
        yVar = MeanSquaredError(Y_test, y_pred_test)
        # The mean squared error of training dataset is stored in the bias array
        bias.append(yBias)
        # The mean squared error of testing dataset is stored in the variance array
        variance.append(yVar)

    return bias, variance


# given the dataset and the betas with the maximum degree until which degree we want the  model complexity to be it returns the nd array of the predicted values of the model
def model(
    xplot: list[float], betaArray: list[list[float]], deg: int
) -> list[list[float]]:
    return np.array(
        [
            sum(betaArray[i][j] * (xplot**j) for j in range(len(betaArray[i])))
            for i in range(deg + 1)
        ]
    )


def main():
    # Generating 100 evenly spaced numbers between -5 and 5
    # contains the whole population of data
    X_init = np.linspace(-5, 5, 100)

    # Calling the datasetGenerator function to get y for every x in X
    Y_init = datasetGenerator(X_init)

    XYtup = []
    for i in range(len(X_init)):
        XYtup.append(tuple([X_init[i], Y_init[i]]))

    # shuffling the list of tuples using inplace shuffling method in numpy random shuffl
    np.random.shuffle(np.array(XYtup))

    # getting the training dataset
    # The X array is sliced to get the training array, this is also the X^1 array
    X = np.array([XYtup[i][0] for i in range(len(X_init))])
    Y = np.array([XYtup[i][1] for i in range(len(X_init))])

    # Splitting the dataset into training, validation, and testing sets
    split_idx_train = int(X.shape[0] * 0.8)
    split_idx_test = int(X.shape[0] - (X.shape[0] * 0.8))

    # creating the test and training part of the dataset
    X_train, Y_train = zip(*(random.sample(XYtup, split_idx_train)))
    X_test, Y_test = zip(*(random.sample(XYtup, split_idx_test)))

    # finding the training and test error for the lagrange model
    Y_train_lag = np.array(
        [
            lagrangeInterpolation(X_train, Y_train, X_train[i], len(X_train))
            for i in range(len(X_train))
        ]
    )
    Y_test_lag = np.array(
        [
            lagrangeInterpolation(X_train, Y_train, X_test[i], len(X_train))
            for i in range(len(X_test))
        ]
    )

    # x plot is a linspace between -5 and 5 with 10000 elements in between
    xplot = np.linspace(-5, 5, 10000)
    deg = 10

    # this array contains the beta values till degree of 10
    betaArray = [betaCalculation(X_train, Y_train, i) for i in range(deg + 1)]

    # contains the x values in linspace of -5 to 5 with 1000 values
    models = model(xplot, betaArray, deg)

    bias, variance = biasVarianceTradeoff(
        X_train, Y_train, X_test, Y_test, deg, betaArray
    )

    # print the variance of bias of models of degree 1 - 10 with lagrange
    print(f"the bias of model with degree: {deg} is {bias}")
    print(f"the bias of model with degree: {deg} is {variance}")
    print(f"the bias of legrange's modes {Y_train_lag} and variance is {Y_test_lag}")

    # plotting the predictions and the actual values of the models
    for i in range(1, deg + 1):
        if i == 1:
            plt.plot(xplot, models[i], label=f"{i}st degree model ")
        elif i == 2:
            plt.plot(xplot, models[i], label=f"{i}nd degree model ")
        elif i == 3:
            plt.plot(xplot, models[i], label=f"{i}rd degree model ")
        else:
            plt.plot(xplot, models[i], label=f"{i}th degree model ")

    # using scatterplot to plot the actual values
    plt.scatter(X_train, Y_train, label="Actual values", marker="x")
    plt.title(f"Model performance from degree 1 to {deg}")
    plt.xlabel(f"feature - X, X**2 ... , X ** {deg}")
    plt.ylabel("model prediction")
    plt.figtext(
        0.5,
        0.01,
        f"Model performance from degree 1 to {deg}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
    )
    plt.legend()
    plt.show()
    plt.close()

    # Plotting the bias-variance trade-off
    plt.plot([i for i in range(0, deg + 1)], bias, c="b", label="Bias", marker=".")
    plt.title("Bias-Variance Trade off")
    plt.xlabel("Model Complexity")
    plt.ylabel("Error")
    plt.figtext(
        0.5,
        0.01,
        f"This plot represent the bias, variance trade off when we increase the model complexity from degree {1} to {deg+1}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
    )
    # iterating from 1 to degree + 1 to print the variance
    plt.plot(
        [i for i in range(0, deg + 1)], variance, c="r", label="Variance", marker="."
    )
    plt.title("Bias-Variance Trade off")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
