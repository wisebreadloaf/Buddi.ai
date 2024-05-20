# random library used for random sampling
import random

# import math library
import math

# Importing the numerical Python library
import numpy as np

# Importing matplotlib for plotting the graph
import matplotlib.pyplot as plt


# Function to generate a dataset based on the input x
def datasetGenerator(x: int) -> int:
    return ((2 * x) - 3) + np.random.normal(0, 5)


# mean squared error calculation for comparing the actual y with the prediction
def meanSquaredError(y: float, y_pred: float) -> float:
    # summation of squares of all the y with y predictions divided by the length of y using the np.mean method
    # return np.mean((y - y_pred) ** 2)
    return np.mean(np.abs(y - y_pred))


def gradientDescent(
    X: list[float],
    Y: list[float],
    Y_pred: list[float],
    B0: float,
    B1: float,
    learningRate: float,
):
    B0 -= learningRate * np.mean(-2 * (Y - Y_pred))
    B1 -= learningRate * np.mean(2 * (Y - Y_pred) * (-X))
    return B0, B1


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


def stochasticGradient(
    B0: float,
    B1: float,
    X_train: list[float],
    Y_train: list[float],
    X_test: list[float],
    Y_test: list[float],
    learningRate,
) -> tuple[list[list[float]], list[float], list[int]]:
    flag = True
    Yvals = Y_train
    epsTrainArr = []
    epsTestArr = []
    epochsarr = []
    B = []
    epochs = 0

    while flag:
        print(f"alive {epochs}")
        idx = math.floor(np.random.uniform(0, len(X_train)))
        # prediction of the model for previous B0 and B1
        Y_train_pred = B0 + B1 * X_train
        Y_test_pred = B0 + B1 * X_test
        # checking if the code converged or not
        if meanSquaredError(Yvals, Y_train_pred) <= 1e-6:
            # if it converges changing the flag
            flag = False

        # finding the error of the mean squared error
        epsTrain = meanSquaredError(Y_train, Y_train_pred)
        epsTest = meanSquaredError(Y_test, Y_test_pred)

        # appending the arror onto the eps array it has eps until the model converges
        epsTrainArr.append(epsTrain)

        # appending the arror onto the eps array it has eps until the model converges
        epsTestArr.append(epsTest)

        # calculating new B0 and B1 using gradient descent
        B0, B1 = gradientDescent(
            X_train[idx], Y_train[idx], Y_train_pred[idx], B0, B1, learningRate
        )
        B.append([B0, B1])
        Yvals = Y_train_pred

        # adding the epoch value to see how it converges
        epochs += 1

        # appending the apochs onto the epochs array it has epoch count until the model converges
        epochsarr.append(epochs)

    return B, epsTrainArr, epsTestArr, epochsarr


# gets executed when the file is executed
def main():
    X_init = np.array(np.linspace(-5, 5, 1000))
    # the population is generated using the populationGenerator function for the whole input feature array X and returns an array of the population
    Y_init = [datasetGenerator(X_init[i]) for i in range(len(X_init))]

    XYtup = []
    for i in range(len(X_init)):
        # creating a list of tuples => [(X,Y)]
        XYtup.append(tuple([X_init[i], Y_init[i]]))
    # inplace shuffling of the tuples
    np.random.shuffle(XYtup)

    # getting the training feature values and output
    X = np.array([XYtup[i][0] for i in range(len(X_init))])
    Y = np.array([XYtup[i][1] for i in range(len(X_init))])

    # The index for the first 80-20 split is computed
    split_idx = int(X.shape[0] * 0.8)

    # A random sample is extracted off the XYtup, which is a list of tuples of the size of split_idx
    train_data = random.sample(XYtup, split_idx)
    # The data which is not in train_data but is in XYtup is utilized as the test data
    test_data = [i for i in XYtup if i not in train_data]
    # The X_test and Y_test value is extracted from the list of tuples test_data
    X_test, Y_test = zip(*test_data)
    # The X_test and Y_test value is extracted from the list of tuples test_data
    X_train, Y_train = zip(*train_data)
    # The learning rate is initialized

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # calculating the B0 and B1 with normal random
    B0 = np.random.normal(0, 1)
    B1 = np.random.normal(0, 1)
    # putting the B0 and B1 in a list
    B = [[B0, B1]]
    # learningRate = 0.0015625
    learningRate = 0.001

    # while learningRate <= 0.1:
    B, epsTrainArr, epsTestArr, epochsarr = stochasticGradient(
        B0, B1, X_train, Y_train, X_test, Y_test, learningRate
    )
    epochs = epochsarr[-1]

    # printing the Beta computed using gradient descent and closedform solution
    print(f"B0 and B1 after the model converges: {B[-1]} with error {epsTrainArr[-1]}")
    print(epsTestArr[-1])
    print(f"B0 and B1 for closed form solution: {betaCalculation(X, Y, 1)}")

    # printing the epoch value to see how it converges
    print(
        f"number of epochs needed for convergence: {epochs} for learning rate: {learningRate}\n"
    )
    plt.plot(
        list(range(100, epochs)),
        epsTrainArr[100:],
        label=f"Epoch vs Training Error for learning rate: {learningRate}",
    )
    plt.plot(
        list(range(100, epochs)),
        epsTestArr[100:],
        label=f"Epoch vs Testing Error for learning rate: {learningRate}",
    )

    # learningRate *= 2
    print(len(B))

    # setting the title of the graph to specify which axis has which variable and other things
    plt.title("The epoch count vs the epsilon of the model with stochastic gradient")
    plt.xlim(0, 5000)
    # plotting with the x label as epochs
    plt.ylabel("Error")
    # plotting with the y label as error
    plt.xlabel("Epochs")
    # having a description for the graph to explain what it does
    plt.figtext(
        0.5,
        0.01,
        f"this plot represents how the mean squared error of the model changes for testing and training as the number of epoch increases for stochastic gradient",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
        bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
    )
    # legends to explain which coloured line represents which learning rate
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
