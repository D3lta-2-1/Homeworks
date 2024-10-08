import matplotlib.pyplot as plt
import numpy as np
import csv

def open_csv():
    with open("data2.csv", newline="") as csvfile:
        data = list(csv.reader(csvfile))
       
    to_float = lambda x: np.array(x).astype(float)
    return to_float(data[0]), to_float(data[1]), to_float(data[2]), to_float(data[3])

def analyse(U, I):
    average = np.average(U / I)
    standard_deviation = np.std(U / I)
    print("average:", average, "standard deviation:", standard_deviation)

def modelise(x, y):
    #linear regression
    [a, b] = np.polyfit(x, y, 1) # shearch for a polynomial of degree 1 that fits the data
    x_model = np.linspace(x[0], x[len(x) - 1], 10) #generate 100 points between 0 and 5
    y_model = a * x_model + b
    return x_model, y_model

def regression_monto_carlo(U, u_U, I, u_I):
    N = 10000
    R_sim = np.zeros(N)

    a = []
    b = []


    for i in range(N):
        delta_U = np.random.uniform(-u_U, u_U)
        delta_I = np.random.uniform(-u_I, u_I)
        [a_sim, b_sim] = np.polyfit(U + delta_U, I + delta_I, 1)
        a.append(a_sim)
        b.append(b_sim)
    
    a = np.array(a)
    b = np.array(b)
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    x_model = np.linspace(0, U[len(U) - 1], 10)
    y_model = a_mean * x_model + b_mean

    print("mean \n\t a:", a_mean, "b:", b_mean)
    print("standard deviation \n\t a:", np.std(a), "b:", np.std(b))

    return x_model, y_model

def main():
    U, u_U, I, u_I = open_csv()
    analyse(U, I)

    plt.errorbar(U, I, xerr=u_U *2, yerr=u_I *2, fmt='r+')

    #x_model, y_model = analyse(U, I)
    #x_model, y_model = modelise(U, I)
    #plt.plot(x_model, y_model, 'b')

    plt.xlabel('U')
    plt.ylabel('I')
    plt.title('I = f(U)')
    plt.grid()
    plt.show()
    
    return

if __name__ == "__main__":
    main()