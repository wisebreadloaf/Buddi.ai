import numpy as np
import matplotlib.pyplot as plt


txt1 = "The width of the plot changes with different standard deviations"
txt2 = "The height and the position of peak of the plot changes with different means"


def normalDistribution(x:float , mean: float , stdev:float):
    return (1 / (stdev * np.sqrt(2 * np.pi))) * np.exp(
        -1 / 2 * (((x - mean) / stdev) ** 2)
    )


mean = 1
stdev1, stdev2, stdev3 = 15, 25, 35
X1 = np.arange(mean - 3 * stdev1, mean + 3 * stdev1, 0.01)
X2 = np.arange(mean - 3 * stdev2, mean + 3 * stdev2, 0.01)
X3 = np.arange(mean - 3 * stdev3, mean + 3 * stdev3, 0.01)
Y1 = normalDistribution(X1, mean, stdev1)
Y2 = normalDistribution(X2, mean, stdev2)
Y3 = normalDistribution(X3, mean, stdev3)

plt.figure(figsize=(10, 6))
plt.plot(X1, Y1, label=f"mean(μ): {mean} Standard deviation(δ): {stdev1}")
plt.plot(X2, Y2, label=f"mean(μ): {mean} Standard deviation(δ): {stdev2}")
plt.plot(X3, Y3, label=f"mean(μ): {mean} Standard deviation(δ): {stdev3}")
plt.legend()
plt.title("Gaussian distributions: Same mean different standard deviation")
plt.xlabel("X")
plt.ylabel("Y")
plt.figtext(
    0.5,
    0.01,
    txt1,
    wrap=True,
    horizontalalignment="center",
    fontsize=8,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.grid(True)
plt.show()

stdev = 1
mean1, mean2, mean3 = 1, 2, 3
X1 = np.arange(mean1 - 3 * stdev, mean1 + 3 * stdev, 0.1)
X2 = np.arange(mean2 - 3 * stdev, mean2 + 3 * stdev, 0.1)
X3 = np.arange(mean3 - 3 * stdev, mean3 + 3 * stdev, 0.1)
Y1 = normalDistribution(X1, mean1, stdev)
Y2 = normalDistribution(X2, mean2, stdev)
Y3 = normalDistribution(X3, mean3, stdev)

plt.figure(figsize=(10, 6))
plt.plot(X1, Y1, label=f"mean(μ): {mean1} Standard deviation(δ): {stdev}")
plt.plot(X2, Y2, label=f"mean(μ): {mean2} Standard deviation(δ): {stdev}")
plt.plot(X3, Y3, label=f"mean(μ): {mean3} Standard deviation(δ): {stdev}")
plt.legend()
plt.title("Gaussian distributions: Same standard deviation different mean")
plt.xlabel("Feature values")
plt.ylabel("Density values")
plt.figtext(
    0.5,
    0.01,
    txt2,
    wrap=True,
    horizontalalignment="center",
    fontsize=8,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.grid(True)
plt.show()