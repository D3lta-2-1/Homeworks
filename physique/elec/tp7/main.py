import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import itertools as it

R = 28 #Ohm

def read_csv(filename):
    data  = pd.read_csv(filename, delimiter=",")
    
    #Temps,EA04_D ,EA15_D,EA26_D,EA37_D
    
    time = data["Temps"].to_numpy()
    bobine = data['EA04_D'].to_numpy()
    resistor = data['EA15_D'].to_numpy()
    gbf = data['EA26_D'].to_numpy()
    condensator = data['EA37_D'].to_numpy()
    
    return time, gbf, condensator, -bobine, resistor

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
    return start + 0, end #ignore imprecision at start

def main():
    time, gbf, condensator, bobine, resistor = read_csv(sys.argv[1])

    #normalize GBF
    mean = np.mean(gbf)
    gbf = gbf - mean
    start, end = find_slice_and_ignore(gbf, -1)
    
    #get a discharge
    gbf = gbf[start:end]
    time = time[start:end]
    condensator = condensator[start:end]
    bobine = bobine[start:end]
    resistor = resistor[start:end]

    #cancel noise
    bobine = bobine - (gbf - np.mean(gbf))
    gbf = np.array([0] * len(gbf))


    i = resistor / R
    du = np.gradient(condensator) / np.diff(time) #derivee de u	
    c = (i[0:-1] / du)
    print(np.mean(c))

    #Plot 
    plt.title("tout un bordel")
    plt.plot(time, gbf, 'r-')
    plt.plot(time, resistor, 'g-')
    plt.plot(time, condensator, 'y-')
    plt.plot(time, bobine, 'b-')
    

    #plt.plot(time, resistor, 'y')

    #plt.subplot(2, 2, 1)
    #plt.subplot(2, 2, 2)
    #plt.subplot(2, 2, 3)/
    #plt.subplot(2, 2, 4)

    plt.show()

if __name__ == "__main__":
    main()