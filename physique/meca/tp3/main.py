import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it

from pandas.core.generic import dt

def read_csv(filename):
    data  = pd.read_csv(filename, delimiter=",")

    time = data["Temps"].to_numpy()
    x = data["MouvementX2"].to_numpy()
    y = data["MouvementY2"].to_numpy()

    return time, x, y

def calculate_angle(x, y):
    return (np.arctan2(y, x), np.sqrt(x**2 + y**2))

def energie_cinetique(m, d_theta, r):
    return 0.5 * m * (r * d_theta)**2

def energie_potentielle(m, g, theta, r):
    return m * g * r * (1 - np.cos(theta))

def main():
    time, x, y = read_csv("bonneboule.csv")

    plt.subplot(221)
    plt.plot(x, y, 'b+', label="y=f(x)")

    plt.subplot(222)
    theta, r = calculate_angle(x, y)
    theta = theta - np.mean(theta)

    d_theta = np.diff(theta) / np.diff(time)

    plt.plot(time, theta, 'r+', label="theta")

    cinetique = energie_cinetique(0.117, d_theta, 0.174)
    potentielle = energie_potentielle(0.117, 09.81, theta, 0.174)

    plt.subplot(223)
    plt.plot(time[1:], cinetique, 'b-', label="d_theta")
    plt.plot(time, potentielle, 'r-', label="theta")
    plt.plot(time[1:], cinetique + potentielle[1:], 'g-', label="total")
    plt.show()


if __name__ == "__main__":
    main()
