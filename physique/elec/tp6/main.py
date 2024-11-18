import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def read_data(filename):
    data = pd.read_csv(filename)
    time = np.array(data["Temps"])
    pos_x = np.array(data["Mouvement X"])
    pos_y = np.array(data["Mouvement Y"])
    return time, pos_x, pos_y

def gradient(coord, time):
    return np.gradient(coord, time)

def gradient_vector(pos_x, pos_y, time, order):
    for _ in range(order):
        pos_x = gradient(pos_x, time)
        pos_y = gradient(pos_y, time)
    return pos_x, pos_y

def lenght_and_angle(x, y):
    length = np.sqrt(x**2 + y**2)
    angle = np.arccos(x/length) * np.sign(y)
    angle = np.degrees(angle)
    return length, angle

def main():
    filemane = sys.argv[1]
    time, pos_x, pos_y = read_data(filemane)

    #vecteur vitesse
    vx, vy = gradient_vector(pos_x, pos_y, time, 1)
    #vecteur acceleration
    ax, ay = gradient_vector(vx, vy, time, 1)

    #vitesse initiale
    length, angle = lenght_and_angle(vx[0], vy[0])

    #acceleration moyenne
    axm = np.mean(ax)
    aym = np.mean(ay)
    g = np.sqrt(axm**2 + aym**2)

    # traduction en coordonnées polaires
    r, theta = lenght_and_angle(pos_x, pos_y)

    #affichage
    plt.subplot(2, 2, 1)
    plt.plot(time, pos_y, label="Trajectoire y", color="blue")
    plt.plot(time, pos_x, label="Trajectoire x", color="red")

    plt.title("Trajectoire")
    plt.xlabel("Temps")
    plt.ylabel("Position")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.quiver(pos_x, pos_y, vx, vy, label="Vitesse\nV0 = {:.3}, alpha = {:.3}".format(length, angle), color="red")
    plt.quiver(pos_x, pos_y, ax, ay, label="Acceleration\ng = {:.3}".format(g), color="green")
    plt.title("Vitesse et acceleration")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.legend(loc = "best")
    plt.grid()
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(time, r, label="Trajectoire r", color="blue")
    plt.title("R en fonction du temps")
    plt.grid()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(time, theta, label="Trajectoire theta", color="red")
    plt.title("Theta en fonction du temps")
    plt.grid()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()