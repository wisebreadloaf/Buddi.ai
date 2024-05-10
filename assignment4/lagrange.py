# importing numerical python library
import numpy as np

# importing matplotlib for plotting the graph
import matplotlib.pyplot as plt


# datasetGenerator for generating y for every x in X
def datasetGenerator(x: int) -> int:
    return (
        ((2 * x) ** 4)
        - ((3 * x) ** 3)
        + ((7 * x) ** 2)
        + (23 * x)
        + 8
        # for adding random noise
        + np.random.normal(0, 3)
    )


# equally seperated 100 values between -5 and 5
X = np.linspace(-5, 5, 100)
# passing the values through the datasetGenerator to get the Y array with y for every x in X.
Y = datasetGenerator(X)

# X1 is X**1 which is equal to X
X1 = X
# X0 is to be multiplied with B0 when doing the matmul hence X1**0 which is equal to 1
X0 = np.array(X1**0)
# X2 is a new feature created from X by doing X ** 2
X2 = np.array(X1**2)
# X3 is a new feature created from X by doing X ** 3
X3 = np.array(X1**2)
# X4 is a new feature created from X by doing X ** 4
X4 = np.array(X1**2)

# initialising the size of the training set
split_idx_train = int(X.shape[0] * 0.7)

# initialising the size of the validation set
split_idx_valid = split_idx_train + int(X.shape[0] * 0.1)

# training split of X from X0 to X4
X1_train = X[:split_idx_train]
X0_train = np.array(X1_train**0)
X2_train = np.array(X1_train**2)
X3_train = np.array(X1_train**3)
X4_train = np.array(X1_train**4)

# getting the Y_train for every x in X1_Train
Y_train = datasetGenerator(X1_train)

# validation split of X from X0 to X4
X1_valid = X[:split_idx_valid]
X0_valid = np.array(X1_valid**0)
X2_valid = np.array(X1_valid**2)
X3_valid = np.array(X1_valid**3)
X4_valid = np.array(X1_valid**4)

# getting the Y_train for every x in X1_Train
Y_valid = datasetGenerator(X1_valid)

# validation split of X from X0 to X4
X1_test = X[split_idx_valid:]
X0_test = np.array(X1_test**0)
X2_test = np.array(X1_test**2)
X3_test = np.array(X1_test**3)
X4_test = np.array(X1_test**4)

# getting the Y_train for every x in X1_Train
Y_test = np.array(datasetGenerator(X1_test))


# X_leg = np.array([X0_test, X1_test, X2_test, X3_test, X4_test])
# X_fourT = np.transpose(X_leg)

# array of elements of X0_test till X4_test
X_tranpose = np.array([X0_test, X1_test, X2_test, X3_test, X4_test])

# transpose of array of elements this is the original array.
X_four = np.transpose(X_tranpose)

# dot product of X transpose of X
XTX = np.matmul(X_tranpose, X_four)

# inverse of the X transpose and X
XTXm1 = np.linalg.inv(XTX)

# dot product of the inverse of dot product X transpose and X with the transpose of the array
XTXintoXT = np.matmul(XTXm1, X_tranpose)
B_four = np.matmul(XTXintoXT, Y_test)


# modeling the linear polynomial equation with B0 and B1
def linearModel(X1: list[float], B_four: list[float]) -> list[float]:
    return B_four[0] + (B_four[1] * X1)


# modeling the quadratic polynomial equation with B0, B1 and B2
def quadraticModel(
    X1: list[float], X2: list[float], B_four: list[float]
) -> list[float]:
    return B_four[0] + ((B_four[1] * X1) + (B_four[2] * X2))


# modeling the quadratic polynomial equation with B0, B1, B2 and B3
def CubicModel(
    X1: list[float], X2: list[float], X3: list[float], B_four: list[float]
) -> list[float]:
    return B_four[0] + ((B_four[1] * X1) + (B_four[2] * X2) + (B_four[3] * X3))


# modeling the quadratic polynomial equation with B0, B1, B2, B3 and B4
def quarternaryModel(
    X1: list[float],
    X2: list[float],
    X3: list[float],
    X4: list[float],
    B_four: list[float],
) -> list[float]:
    return B_four[0] + (
        (B_four[1] * X1) + (B_four[2] * X2) + (B_four[3] * X3) + (B_four[4] * X4)
    )


# lagrange interpolation to fit the equation into a curve
def lagrangeInterpolation(x_values, y_values, x, n):
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


# writing a function to return the outputs of the all the polynomial models defined before.
def models(
    X1: list[float],
    X2: list[float],
    X3: list[float],
    X4: list[float],
    B_four: list[float],
) -> tuple[float]:
    lin = linearModel(X1, B_four)
    quad = quadraticModel(X1, X2, B_four)
    cube = CubicModel(X1, X2, X3, B_four)
    quart = quarternaryModel(X1, X2, X3, X4, B_four)

    # fitting the inputs and outputs with langrange polynomial model
    lagrange = [lagrangeInterpolation(X1, Y, x, len(X1)) for x in X1]

    # returning the polynomial functions
    return [lin, quad, cube, quart, lagrange]


models = models(X1, X2, X3, X4, B_four)


# writing a function to send the input data through the training set to all the models
def train(X1_train):
    lin_train = linearModel(
        X1_train, B_four
    )  # the output array after running the input training array and betas into the linearModel function is returned
    quad_train = quadraticModel(
        X1_train, X2_train, B_four
    )  # the output array after running the input training array and betas into the quaratic Model function is returned
    cube_train = CubicModel(
        X1_train, X2_train, X3_train, B_four
    )  # the output array after running the input training array and betas into the cubic Model function is returned
    quart_train = quarternaryModel(
        X1_train, X2_train, X3_train, X4_train, B_four
    )  # the output array after running the input training array and betas into the quarternary Model function is returned
    lagrange_train = [
        lagrangeInterpolation(X1_train, Y_train, x, len(X1_train)) for x in X1_train
    ]  # the output array after running the input training array and betas into the lagrangeModel function is returned

    # epsilon of the linear model with the X_train
    eps_lin_train = np.sum(np.abs(Y_train - lin_train) / len(X1_train))

    # epsilon of the quadratic model with the X_train
    eps_quad_train = np.sum(np.abs(Y_train - quad_train) / len(X1_train))

    # epsilon of the linear model with the X_train
    eps_cube_train = np.sum(np.abs(Y_train - cube_train) / len(X1_train))

    # the output array after running the input training array and betas into the  function is returned
    eps_quart_train = np.sum(np.abs(Y_train - quart_train) / len(X1_train))
    eps_lagrange_train = np.sum(np.abs(Y_train - lagrange_train) / len(X1_train))

    return [
        eps_lin_train,
        eps_quad_train,
        eps_cube_train,
        eps_quart_train,
        eps_lagrange_train,
    ]


bias_eps = train(X1_train)


def valid(
    X1_valid: list[float],
    X2_valid: list[float],
    X3_valid: list[float],
    X4_valid: list[float],
    B_four: list[float],
):
    lin_valid = linearModel(X1_valid, B_four)
    quad_valid = quadraticModel(X1_valid, X2_valid, B_four)
    cube_valid = CubicModel(X1_valid, X2_valid, X3_valid, B_four)
    quart_valid = quarternaryModel(X1_valid, X2_valid, X3_valid, X4_valid, B_four)
    lagrange_valid = [
    lagrangeInterpolation(X1_test, Y_test, x, len(X1_valid)) for x in X1_valid
    ]

    eps_lin_valid = np.sum(np.abs(Y_valid - lin_valid) / len(X1_valid))
    eps_quad_valid = np.sum(np.abs(Y_valid - quad_valid) / len(X1_valid))
    eps_cube_valid = np.sum(np.abs(Y_valid - cube_valid) / len(X1_valid))
    eps_quart_valid = np.sum(np.abs(Y_valid - quart_valid) / len(X1_valid))
    eps_lagrange_valid = np.sum(np.abs(Y_valid - lagrange_valid) / len(X1_valid))

    # eps_lin_valid = np.sum((Y_valid - lin_valid) ** 2 / len(X1_valid))
    # eps_quad_valid = np.sum((Y_valid - quad_valid) ** 2 / len(X1_valid))
    # eps_cube_valid = np.sum((Y_valid - cube_valid) ** 2 / len(X1_valid))
    # eps_quart_valid = np.sum((Y_valid - quart_valid) ** 2 / len(X1_valid))

    return [
        eps_lin_valid,
        eps_quad_valid,
        eps_cube_valid,
        eps_quart_valid,
        eps_lagrange_valid,
    ]

variance_eps = valid(X1_valid, X2_valid, X3_valid, X4_valid, B_four)
eps_lagrange_valid = variance_eps[4]

variance_eps = variance_eps[:3]
print(variance_eps)
print(bias_eps)

lin_model = models[0]
quad_model = models[1]
cube_model = models[2]
quart_model = models[3]
lagrange_model = models[4]

txt = "Model prediction of Linear,Quadratic,cubic,quaternary,lagrange polynomial models"
plt.scatter(X, Y, label="Actual values", marker="x")
plt.plot(X, lin_model, label="linear polynomial model", marker=".")
plt.plot(X, quad_model, label="quadratic polynomial model", marker=".")
plt.plot(X, cube_model, label="cubic polynomial model", marker=".")
plt.plot(X, quart_model, label="quarternary polynomial model", marker=".")
plt.plot(X, lagrange_model, label="Lagrange polynomial model", marker=".")
plt.title("Model performance")
plt.xlabel("feature - [X]")
plt.ylabel("model prediction")
plt.figtext(
    0.5,
    0.01,
    txt,
    wrap=True,
    horizontalalignment="center",
    fontsize=10,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.legend()
plt.show()
plt.close()


txt = "This plot represent the bias, variance trade off when we increase the model complexity from linear to quadratic to cubic to quarternary and also Lagrange"
x = [1, 2, 3, 4, 70]
plt.plot(x, bias_eps, c="b", label="Bias", marker=".")
plt.title("Bias-Variance Trade off")
plt.xlabel("Model Complexity")
plt.ylabel("Error")
plt.figtext(
    0.5,
    0.01,
    txt,
    wrap=True,
    horizontalalignment="center",
    fontsize=10,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.plot(x[:4], variance_eps, c="r", label="Variance", marker=".")
plt.legend()
plt.show()
