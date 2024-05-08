import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (
        ((2 * x) ** 4)
        - ((3 * x) ** 3)
        + ((7 * x) ** 2)
        + (23 * x)
        + 8
        + np.random.normal(0, 3)
    )


X = np.linspace(-5, 5, 100)
Y = f(X)

# Linear function
X0 = np.array(X**0)
X1 = np.array(X**1)
X2 = np.array(X**2)
X3 = np.array(X**3)
X4 = np.array(X**4)

X_four = np.array([X0, X1, X2, X3])
X_leg = np.array([X0, X1, X2, X3, X4])


XT = np.transpose(X_four)
XTX = np.matmul(XT, X_four)
XTXm1 = np.linalg.inv(XTX)
print(XTXm1.shape)
print(XT.shape)
XTXintoXT = np.matmul(XTXm1, XT)
B_four = np.matmul(XTXintoXT, Y)
