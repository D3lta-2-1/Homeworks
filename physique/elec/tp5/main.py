import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import scipy.signal as signal

RESISTANCE = 1000 #2300 #ohm

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
    for (a, b), i in zip(it.pairwise(data), it.count()):
        if np.sign(a) != np.sign(b):
            if np.sign(b) == kind:
                start = i
            else:
                end = i
                if slice[1] - slice[0] < end - start:
                    slice = (start, end)
    return slice

def find_slice_and_ignore(data, kind):
    start, end = find_slice(data, kind)
    return start + 5, end #ignore imprecision at start

def smooth(data, window_size=20):
    return signal.savgol_filter(data, window_size, 2)

def find_value(data, value, is_growing):
    start = 0
    end = len(data)

    smaller = lambda a, b: a < b
    greater = lambda a, b: a > b

    comp1 = smaller if is_growing else greater
    comp2 = greater if is_growing else smaller

    while True:
        index = (start + end) // 2
        if comp1(data[index], value):
            start = index
        if comp2(data[index], value):
            end = index
        if end - start < 2:
            return index

def find_at_index(value, percent, is_growing):
    start = value[0]
    end = value[-1]
    target_tau = (1 - percent) * start + percent * end
    index = find_value(value, target_tau, is_growing)
    return index

def find_time_at(time, value, percent, is_growing):
    index = find_at_index(value, percent, is_growing)
    return time[index] - time[0], time[index], value[index]

def draw(time, bobine, resistor, generator, title):
    plt.title(title)
    plt.xlabel("Temps (µs)")
    plt.ylabel("Tension (V)")
    plt.grid(True)
    plt.plot(time, generator, 'b+', label="Générateur")
    plt.plot(time, resistor, 'g+', label="Résistance")
    plt.plot(time, bobine, 'r+', label="Bobine")

def draw_raw(time, bobine, resistor, generator):
    plt.subplot(3, 2, 1)
    
    draw(time, bobine, resistor, generator, "Valeur brute")
    plt.legend(loc="best")
    plt.tight_layout()

def draw_smoothed_charging(time, bobine, resistor, generator):
    start, end = find_slice_and_ignore(generator, 1)

    bobine = smooth(bobine[start:end])
    resistor = smooth(resistor[start:end])
    time = time[start:end]
    generator = generator[start:end]

    tau, x, y = find_time_at(time, bobine, 0.63, False)
    plt.subplot(3, 2, 3)
    plt.annotate("tau = {:}µs".format(tau), (x, y))
    draw(time, bobine, resistor, generator, "Mise en tension")
    plt.tight_layout()

def draw_smoothed_decharging(time, bobine, resistor, generator):
    start, end = find_slice_and_ignore(generator, -1)

    bobine = smooth(bobine[start:end])
    resistor = smooth(resistor[start:end])
    generator = generator[start:end]
    tau, x, y = find_time_at(time[start:end], bobine, 0.63, True)

    plt.subplot(3, 2, 5)
    plt.annotate("tau = {:}µs".format(tau), (x, y))
    draw(time[start:end], bobine, resistor, generator, "Régime libre ")
    plt.tight_layout()

def monte_carlo(raw_time, raw_data, d, n_sim):
    array=np.zeros(n_sim)
    for i in range (n_sim):
        simulated_error = np.random.uniform(-d, d, len(raw_data))
        simulated_data = raw_data + simulated_error
        simulated_data = np.log(np.abs(simulated_data))
        a, _ = np.polyfit(raw_time, simulated_data, 1)
        array[i] = -1/a
    
    return np.mean(array), np.std(array)

def draw_log(time, bobine, generator):
    
    start, end = find_slice_and_ignore(generator, 1)

    bobine = smooth(bobine[start:end])
    time = time[start:end]

    plt.subplot(3, 2, 2)
    plt.title("Logarithme des valeurs mesurées")
    log_data = np.log(bobine)
    plt.plot(time, log_data, 'r+')

    index = find_at_index(bobine, 0.95, False) #work on 95% of the charging tension, the rest is noise
    
    time = time[:index]
    log_data = log_data[:index]
    bobine = bobine[:index]

    a, b = np.polyfit(time, log_data, 1)
    
    plt.plot(time, a * time + b, 'b-')
    plt.annotate("1/a = tau = {:}".format(-1/a), (time[-1], log_data[-1]))
    plt.xlabel("""Temps (µs)\n modelisé en bleu\n (95% de la tension)""")
    plt.ylabel("Log de la tension")
    plt.tight_layout()

    start, _ = find_slice(generator, -1)
    d = abs(generator[start + 3]- generator[start + 2])

    N = 10000
    tau, u_tau = monte_carlo(time, bobine, d, N)

    L = tau * 1e-3 * RESISTANCE # 6 would to H but we want mH

    plt.figtext(0,0 ,"En uitilisant la méthode de Monte-Carlo avec Δ = {:.3}V, n = {}\ntau = {:.3} ± {:.3}\nL = {:.3} mH ".format(d, N, tau, u_tau, L))
    return L

def draw_derivative(time, bobine, resistor, generator):
    start, end = find_slice_and_ignore(generator, 1)
    bobine = smooth(bobine[start:end])
    resistor = smooth(resistor[start:end])
    time = time[start:end]

    derivative = np.gradient(resistor, time)

    quotient = bobine / derivative

    plt.subplot(3, 2, 4)
    plt.title("(dUr/dt) / Ub")
    plt.plot(time, quotient, 'r+')
    plt.xlabel("Temps (µs)")

    a, b = np.polyfit(time, quotient, 1)

    plt.plot(time, a * time + b, 'b-')
    plt.text(0, 0, "b = {:}".format(b))
    plt.tight_layout()

def energy(time, bobine, resistor, generator, L):
    start, end = find_slice_and_ignore(generator, 1)
    bobine = smooth(bobine[start:end])
    resistor = smooth(resistor[start:end])
    time = time[start:end]

    i = resistor / RESISTANCE

    e1 = 0.5 * L * 1e-3 * i**2 # mH -> H

    plt.subplot(3, 2, 6)
    plt.title("Energie en fonction du temps")
    plt.plot(time, e1, 'r+')
    plt.xlabel("Temps (µs)")
    plt.ylabel("Energie (J)")
    plt.tight_layout()

def main():
    time, bobine, resistor, generator = read_csv("data.csv")

    time = time * 1e6 #convert to µs
    
    draw_raw(time, bobine, resistor, generator)
    draw_smoothed_charging(time, bobine, resistor, generator)
    draw_smoothed_decharging(time, bobine, resistor, generator)
    L = draw_log(time, bobine, generator)
    draw_derivative(time, bobine, resistor, generator)
    energy(time, bobine, resistor, generator, L)

    plt.show()

if __name__ == "__main__":
    main()