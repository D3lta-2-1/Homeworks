import numpy as np
import numpy.random as rd


def resistance(U, I):
    return U / I


def main():
    U, d_U = 12.0, 0.14
    I, d_I = 0.5, 0.005
    
    N = 100000

    U_sim = rd.uniform(U - d_U, U + d_U, N)
    I_sim = rd.uniform(I - d_I, I + d_I, N)

    R_sim = resistance(U_sim, I_sim)

    e = np.mean(R_sim)
    ecart_type = np.std(R_sim)

    print("Esperance:", e)
    print("Ecart type:", ecart_type)

    return

if __name__ == "__main__":
    main()