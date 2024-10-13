# Merci d'avoir choisi ce Programme pour votre TP4 de physique
# Pour Monsieur HS si vous passez par là, ce programme a bien été écrit par Etienne Thomas a la place de faire son TD de maths

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def lire_csv_oscilloscope(file_path):

    data = pd.read_csv(file_path)
    
    signal_data = {}
    signal_data["time"] = data.iloc[:, 3].to_numpy()
    signal_data["ch1"] = data.iloc[:, 4].to_numpy()
    signal_data["ch2"] = data.iloc[:, 5].to_numpy()

    return signal_data

def smooth(data, window_size=20):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def extractor(time, raw_data, smooth_data, comparator):
    # Find the first index where the smoothed data is greater than the raw data
    charging_time = []
    charging_data = []

    is_charging = not comparator(smooth_data[1], raw_data[0]) #we need to skip the first charge/discharge to know what the initial state is
    charge_start = 0

    for i in range(1, len(raw_data)):
        if comparator(smooth_data[i - 1], smooth_data[i]) and not is_charging:
            is_charging = True
            charge_start = i + 10 #skip the first 3 points to avoid noise
        elif not comparator(smooth_data[i - 1], smooth_data[i]) and is_charging:
            is_charging = False
            charging_time[:0] = map(lambda x: x - time[charge_start], time[charge_start:i])
            charging_data[:0] = raw_data[charge_start:i]
            
    # To keep to smooth working, we fully concatenate the data of multiple charges/discharges        
    sorted_time, sorted_data = zip(*sorted(zip(charging_time, charging_data), key=lambda pair: pair[0]))

    return np.array(sorted_time), np.array(sorted_data)

def charging_model(time, e, tau):
    return e * (1 - np.exp(-time / tau))

def decharging_model(time, e, tau):
    return e * np.exp(-time / tau)

def ajust_model(time, data, model):
    e = max(data)
    tau = 1
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
    file_path = "F0000CH1.CSV"  # Chemin vers le fichier
    signal_data = lire_csv_oscilloscope(file_path)

    plt.title("Signal du GBF")
    plt.xlabel("Temps (s)")
    plt.ylabel("Tension (V)")
    plt.grid(True)

    plt.subplot(3, 1, 1)
    #print raw signal_data
    plt.plot(signal_data["time"], signal_data["ch1"], 'b+',  label='Tension aux bornes du GBF')
    smoothed_voltage = signal.savgol_filter(signal_data["ch1"], 30, 1)
    plt.plot(signal_data["time"], smoothed_voltage, 'r', label='Tension aux bornes du GBF (lissée)')

    #print charging
    plt.subplot(3, 1, 2)

    charging_time, charging_data = extractor(signal_data["time"], signal_data["ch1"], smoothed_voltage, lambda x, y: x < y)
    plt.plot(charging_time, charging_data, 'g+')
    smoothed = signal.savgol_filter(charging_data, 30, 1)
    plt.plot(charging_time, smoothed, 'r', label='Tension aux bornes du GBF (lissée)')
    e, tau = ajust_model(charging_time, charging_data, charging_model)

    print("Charging model: e = ", e, "tau = ", tau)
    modelised_data = charging_model(charging_time, e, tau)
    plt.plot(charging_time, modelised_data, 'b', label='Modèle de charge')


    plt.subplot(3, 1, 3)
    charging_time, charging_data = extractor(signal_data["time"], signal_data["ch1"], smoothed_voltage, lambda x, y: x > y)
    plt.plot(charging_time, charging_data, 'g+')
    smoothed = signal.savgol_filter(charging_data, 30, 1)
    plt.plot(charging_time, smoothed, 'r', label='Tension aux bornes du GBF (lissée)')

    plt.show()

    return

if __name__ == "__main__":
    main()

# Visualisation et mesures sur les données de signal
#visualiser_et_mesurer(signal_data)
