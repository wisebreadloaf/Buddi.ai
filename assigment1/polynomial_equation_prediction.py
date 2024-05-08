import numpy as np

import matplotlib.pyplot as plt

X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
Y1 = np.array([7.0, 2.0, 0.0, 0.0, 0.0, 2.0, 7.0])

# t = list(range(-3, 4))
t = np.arange(-1, 1, 0.1)

t = t.tolist()
weightswtot = []
count = 0
B1 = []
B2 = []
T = []

min_tote = float("inf")
for b1 in t:
    for b2 in t:
        tote = 0
        for x, y in zip(X, Y1):
            y_ = b1 * x + b2 * x**2
            e = abs(y - y_)
            tote += e
        if tote < min_tote:
            min_tote = tote
            min_weights = (b1, b2)
        weightswtot.append((b1, b2, tote))
        B1.append(b1)
        B2.append(b2)
        T.append(tote)

print(min_weights)
print(min_tote)
print(len(B1))
print(len(B2))
print(len(T))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(B1, B2, T, cmap="viridis")

ax.set_title("Input Values and their Minimum Total Error")
ax.set_xlabel("B1")
ax.set_ylabel("B2")
ax.set_zlabel("Total Error")

plt.show()
