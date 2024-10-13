import numpy as np
import numpy.random as rd


def resistance(U, I):
    return U / I


def main():
    U, d_U = 200, 0.1
    I, d_I = 1.6, 0.005

    # intervalle de conf
    racine_3 = np.sqrt(3)
    U_omega = np.sqrt((d_U / racine_3) ** 2 + (d_I / racine_3) ** 2)
    
    N = 10000

    print("U:", U)
    print("d_u:", d_U)
    U_sim = rd.uniform(U - d_U, U + d_U, N)
    I_sim = rd.uniform(I - d_I, I + d_I, N)
    R_sim = resistance(U_sim, I_sim)

    e = np.average(R_sim)
    ecart_type = np.std(R_sim)

    print("Esperance:", e)
    print("Ecart type:", ecart_type)

    return

if __name__ == "__main__":
    main()