import matplotlib.pyplot as plt
import numpy as np
import csv

ANGLE_ERROR = 1.0

def open_csv():
    with open("data.csv", newline="") as csvfile:
        data = list(csv.reader(csvfile))
       
    to_float = lambda x: np.array(x).astype(float)
    #incident, reflechi, d_reflechi, refracté, d_refracté
    return to_float(data[0]), to_float(data[1]), to_float(data[2]), to_float(data[3]), to_float(data[4])

def regression_monto_carlo(x, y, u_x, u_y):
    N = 10000
    R_sim = np.zeros(N)

    a = []
    b = []

    for i in range(N):
        delta_ux = np.random.uniform(-u_x, u_x)
        delta_uy = np.random.uniform(-u_y, u_y)
        [a_sim, b_sim] = np.polyfit(x + delta_ux, y + delta_uy, 1)
        a.append(a_sim)
        b.append(b_sim)
    
    a = np.array(a)
    b = np.array(b)
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    x_model = np.array([0, np.max(x)])
    y_model = a_mean * x_model + b_mean

    print("mean \n\t a:", a_mean, "b:", b_mean)
    print("standard deviation \n\t a:", np.std(a), "b:", np.std(b))

    return x_model, y_model, a_mean, b_mean

def main():
    incident, reflechi, d_reflechi, refracte, d_refracte = open_csv()

    plt.subplot(121)
    plt.xlabel('angle incident')
    plt.ylabel('angle reflechi')
    plt.errorbar(incident, reflechi, xerr=0, yerr=d_reflechi, fmt='r+')  

    print("pour valider la loi sur la reflexion, a doit etre egal à 1")
    x_model, y_model, _, _ = regression_monto_carlo(incident, reflechi, 0, d_reflechi)
    plt.plot(x_model, y_model, 'b')

    plt.title('r = f(i)')
    plt.grid()

    plt.subplot(122)
    sin_incident = np.sin(np.radians(incident))
    sin_refracte = np.sin(np.radians(refracte))
    sin_d_refracte = np.sin(np.radians(d_refracte))
    plt.xlabel('sin angle incident')
    plt.ylabel('sin angle refracté')
    plt.errorbar(sin_incident, sin_refracte, xerr=0, yerr=sin_d_refracte, fmt='r+')  

    print("pour valider la loi sur la refraction, la modelisation doit etre une fonction linéaire")
    print("en supposant que n(air) = 1, n(verre) = 1/a")
    x_model, y_model, a, _ = regression_monto_carlo(sin_incident, sin_refracte, 0, sin_d_refracte)
    print("n(verre) =", 1/a)
    plt.plot(x_model, y_model, 'b')

    plt.title('sin i2 = f(sin(i))')
    plt.grid()
    plt.show()
    
    return

if __name__ == "__main__":
    main()