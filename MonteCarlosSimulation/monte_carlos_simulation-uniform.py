import numpy as np
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

darts_in = 0
tot_darts = 0

log_scale = [
    0,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
]


def estimate_pi(start, end):
    global darts_in, tot_darts
    for j in range(start, end):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)

        if math.sqrt(x**2 + y**2) <= 0.5:
            darts_in += 1
        tot_darts += 1

    return (darts_in / tot_darts) * 4


if __name__ == "__main__":
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        pis = pool.starmap(
            estimate_pi,
            [(log_scale[i], log_scale[i + 1]) for i in range(len(log_scale) - 1)],
        )

    print(pis)

    plt.xscale("log")
    plt.plot(log_scale[1:], pis, label="estimation of Pi", marker=".")
    plt.xlabel("Number of Darts Thrown")
    plt.ylabel("Estimated Pi Value")
    plt.axhline(y=math.pi, color="k", linestyle="--")
    plt.title("Monte Carlo Estimation of Pi using uniform random sampling")
    plt.legend()
    plt.figtext(
        0.5,
        0.01,
        "Monte Carlo simulation to compute the value of pi using repeated uniform random sampling to estimate the probability of dart landing in a circle",
        fontsize=8,
        horizontalalignment="center",
        wrap=True,
        bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
    )

    plt.show()
