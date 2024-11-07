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
    start = 0
    end = 0
    i = 0
    #for a, b in zip(data, data[1:]):
    for a, b, j in it.pairwise(data):
        if np.sign(a) != np.sign(b):
            if np.sign(a) == kind:
                start = i
            else:
                end = i
                return start + 1, end
        i += 1
    return start, end

"""def find_best_slice(data, kind):
    start = 0
    end = 0
    
    while True:
        new_start, new_end, should_break = find_slice(data[end:], kind)
        if new_end - new_start > end - start:
            start = new_start + end
            end = new_end + end
        if should_break:
            break
    return start, end"""

def main():
    time, bobine, resistor, generator = read_csv("data.csv")

    start, end = find_slice(generator, True)

    print(start, end)

    plt.title("test")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)

    plt.plot(time[start:end], bobine[start:end], 'r+', label='Tension aux bornes de la bobine')
    plt.plot(time[start:end], generator[start:end], 'b+', label='Tension aux bornes du générateur')
    plt.plot(time[start:end], resistor[start:end], 'g+', label='Tension aux bornes de la résistance')

    plt.show()

if __name__ == "__main__":
    main()