# Merci d'avoir choisi ce Programme pour votre TP4 de physique
# Pour Monsieur HS si vous passez par là, ce programme a bien été écrit par Etienne Thomas a la place de faire son TD de maths

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def lire_csv_oscilloscope(ch1, ch2):

    data  = pd.read_csv(ch1)
    
    signal_data = {}
    signal_data["time"] = data.iloc[:, 3].to_numpy()
    signal_data["ch1"] = data.iloc[:, 4].to_numpy()

    data = pd.read_csv(ch2)
    signal_data["ch2"] = data.iloc[:, 4].to_numpy()

    return signal_data

def smooth(data, window_size=20):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# extractor 1 isn't super robust
def extractor(time, raw_data, smooth_data, comparator):
    # Find the first index where the smoothed data is greater than the raw data
    charging_time = []
    charging_data = []

    is_charging = comparator(smooth_data[1], raw_data[0]) #we need to skip the first charge/discharge to know what the initial state is
    charge_start = 0

    for i in range(1, len(raw_data)):
        if comparator(smooth_data[i - 1], smooth_data[i]) and not is_charging:
            is_charging = True
            charge_start = i + 30 #skip the size of the smooth
        elif not comparator(smooth_data[i - 1], smooth_data[i]) and is_charging:
            is_charging = False
            charging_time[:0] = map(lambda x: x - time[charge_start], time[charge_start:i])
            charging_data[:0] = raw_data[charge_start:i]
            sorted_time, sorted_data = zip(*sorted(zip(charging_time, charging_data), key=lambda pair: pair[0]))
            return np.array(sorted_time), np.array(sorted_data)
            
    # To keep to smooth working, we fully concatenate the data of multiple charges/discharges        
    sorted_time, sorted_data = zip(*sorted(zip(charging_time, charging_data), key=lambda pair: pair[0]))

    return np.array(sorted_time), np.array(sorted_data)

#extractor 2
def extractor2(time, raw_data, generator_data, keep_it):
    charging_time = []
    charging_data = []
    last_keep_it = keep_it(generator_data[0])

    for (time_point, raw, gen) in zip(time, raw_data, generator_data):
        if keep_it(gen) and last_keep_it != keep_it(gen):
            charging_time.append(time_point)
            charging_data.append(raw)
        last_keep_it = keep_it(gen)

    # To keep to smooth working, we fully concatenate the data of multiple charges/discharges        
    sorted_time, sorted_data = zip(*sorted(zip(charging_time, charging_data), key=lambda pair: pair[0]))

    return np.array(sorted_time), np.array(sorted_data)


def charging_model(time, e, tau):
    return e * (1 - np.exp(-time / tau))

def decharging_model(time, e, tau):
    return e * np.exp(-time / tau)

def cost(time, data, e, tau, model):
    modelised_data = model(time, e, tau)
    distance = modelised_data - data
    distance = distance**2
    return 

def ajust_model(time, data, model):
    e = max(data) - min(data)
    tau = 1
    last_cost = 1000
    last_step = 2
    while True:
        break


    for i in range(1000):
        modelised_data = model(time, e, tau)
        distance = modelised_data - data
        distance = distance**2
        if np.sum(distance) < 0.001:
            return e, tau
        tau -= 0.001
    print("Ajustement failed")
    return e, tau


def main():
    signal_data = lire_csv_oscilloscope("F0002CH1.CSV", "F0002CH2.CSV")

    plt.title("Signal du GBF")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)

    plt.subplot(3, 1, 1)
    #print raw signal_data
    plt.plot(signal_data["time"], signal_data["ch1"], 'b+',  label='Tension aux bornes du GBF')
    plt.plot(signal_data["time"], signal_data["ch2"], 'g',  label='Tension aux bornes du generateur')
    smoothed_voltage = signal.savgol_filter(signal_data["ch1"], 30, 1)
    plt.plot(signal_data["time"], smoothed_voltage, 'r', label='Tension aux bornes du GBF (lissée)')

    #print charging
    plt.subplot(3, 1, 2)

    charging_time, charging_data = extractor(signal_data["time"], signal_data["ch1"], smoothed_voltage, lambda x, y: x < y)
    plt.plot(charging_time, charging_data, 'g+')
    smoothed = charging_data #signal.savgol_filter(charging_data, 2, 1)
    plt.plot(charging_time, smoothed, 'r', label='Tension aux bornes du GBF (lissée)')
    e, tau = ajust_model(charging_time, charging_data, charging_model)

    print("Charging model: e = ", e, "tau = ", tau)
    modelised_data = charging_model(charging_time, e, tau)
    plt.plot(charging_time, modelised_data, 'b', label='Modèle de charge')


    #discharge is borken for now
    plt.subplot(3, 1, 3)
    charging_time, charging_data = extractor(signal_data["time"], signal_data["ch1"], smoothed_voltage, lambda x, y: x > y)
    plt.plot(charging_time, charging_data, 'g+')
    smoothed = charging_data #signal.savgol_filter(charging_data, 30, 1)
    plt.plot(charging_time, smoothed, 'r', label='Tension aux bornes du GBF (lissée)')

    plt.show()

    return

if __name__ == "__main__":
    main()
