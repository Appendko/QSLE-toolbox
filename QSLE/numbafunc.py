import numpy as np
from numba import njit, complex128, float64

# JIT Functions for tensor products
@njit
def H_Psi_Numba(H, Psi):
    # C_{ij} += H_{ijk} \psi{ik}    
    Ni, Nj, Nk = H.shape
    C = np.zeros((Ni, Nj),dtype=np.complex128)  
    for i in range(Ni):
        for j in range(Nj):
            for k in range(Nk):
                C[i, j] += H[i][j][k] * Psi[i][k]    
    return C

@njit
def expectation_Numba(Operator: complex128[:,:,:], Psi: complex128[:,:]): #mjk, ik
    Nm, Nj, Nk = Operator.shape
    Ni = Psi.shape[0]
    res = np.zeros((Ni, Nm), dtype=np.complex128)  
    denominator = np.zeros((Ni,), dtype=np.float64)
    for i in range(Ni):
        for k in range(Nk):
            for m in range(Nm):
                for j in range(Nj):                        
                    res[i, m] += Operator[m][j][k] * np.conj(Psi[i][j]) * Psi[i][k]
    for i in range(Ni):
        for k in range(Nk):
            denominator[i] += (np.conj(Psi[i][k]) * Psi[i][k]).real                    
    for i in range(Ni):
        for m in range(Nm):
            res[i, m] /= denominator[i]
    return res

@njit
def F_Numba(x_tensor, gamma_t, ev_x_t): #mjk, im, im
    Nm, Nj, Nk, = x_tensor.shape
    Ni = gamma_t.shape[0]
    D = np.zeros((Ni, Nj, Nk), dtype=np.complex128)  
    for i in range(Ni):    
        for m in range(Nm):
            for j in range(Nj):
                for k in range(Nk):
                    D[i, j, k] += x_tensor[m][j][k] * gamma_t[i][m]
                D[i, j, j] -= gamma_t[i][m] * ev_x_t[i][m] # * id_N[j][j] == 1 
    return D
