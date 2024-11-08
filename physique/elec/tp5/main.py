import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def read_csv(filename):
    data  = pd.read_csv(filename, delimiter=",")
    
    time = data["Temps"].to_numpy()
    bobine = data['EA04_D'].to_numpy()
    resistor = data['EA15_D'].to_numpy()
    generator = data['EA26_D'].to_numpy()

    return time, bobine, resistor, generator


def find_slice(data, kind):
    slice = (0, 0)
    start = 0
    i = 0
    for a, b in it.pairwise(data):
        if np.sign(a) != np.sign(b):
            if np.sign(b) == kind:
                print("find start at", i)
                start = i
            else:
                end = i
                print("find end", i)
                if slice[1] - slice[0] < end - start:
                    slice = (start, end)
                    print("new best slice", slice)
        i += 1
    return slice

def find_slice_and_ignore(data, kind):
    start, end = find_slice(data, kind)
    return start + 2, end #ignore imprecision at start

def main():
    time, bobine, resistor, generator = read_csv("data.csv")

    plt.subplot(1, 3, 1)

    plt.title("Valeur brute")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)
    plt.plot(time, bobine, 'r+', label='Tension aux bornes de la bobine')
    plt.plot(time, generator, 'b+', label='Tension aux bornes du générateur')
    plt.plot(time, resistor, 'g+', label='Tension aux bornes de la résistance')

    plt.subplot(1, 3, 2)
    
    start, end = find_slice_and_ignore(generator, 1)
    print(start, end)
    plt.title("Mise en tension")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)
    plt.plot(time[start:end], bobine[start:end], 'r+', label='Tension aux bornes de la bobine')
    plt.plot(time[start:end], generator[start:end], 'b+', label='Tension aux bornes du générateur')
    plt.plot(time[start:end], resistor[start:end], 'g+', label='Tension aux bornes de la résistance')

    plt.subplot(1, 3, 3)

    start, end = find_slice_and_ignore(generator, -1)
    print(start, end)
    plt.title("Régime libre")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)
    plt.plot(time[start:end], bobine[start:end], 'r+', label='Tension aux bornes de la bobine')
    plt.plot(time[start:end], generator[start:end], 'b+', label='Tension aux bornes du générateur')
    plt.plot(time[start:end], resistor[start:end], 'g+', label='Tension aux bornes de la résistance')


    plt.show()

if __name__ == "__main__":
    main()