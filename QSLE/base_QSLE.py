import numpy as np
from QSLE import H_Psi_Numba, expectation_Numba, F_Numba
from QSLE.common import DEBYE_SI, DIP_AU_SI, DEBYE2AU, T_FS2AU, WN2AU

class base_QSLE():
    """    
    QSLE Classes: Define the common interface of for 1D and 2D IR Spectra calculations
    """
    def __init__(self, param):
        self.param = param
        self.H = None
        self.H_aux = None
        self.x_tensor = None
        self.p_tensor = None
        self.X_mu = None
        self.Psi_arr = None
        self.V = None           
        self.H_size = None
        self.N_mode = None
        self.U = None
        self.worker_size, self.worker_id = 1, 0

        # Parameters Defined in 1D/2D sub-classes
        self.aux_size = None

        # Constants Defined after updating params        
        self.c_env = None
        self.dt = None
        return

    def propRK4_lambda(self, Psi_i, H_lambda):
        H_mat = H_lambda(Psi_i) # Evaluate H only with current Psi_i
        k1 = (-1j) * self.dt * (H_mat @ (Psi_i           )).ravel()
        k2 = (-1j) * self.dt * (H_mat @ (Psi_i + 0.5 * k1)).ravel()
        k3 = (-1j) * self.dt * (H_mat @ (Psi_i + 0.5 * k2)).ravel()
        k4 = (-1j) * self.dt * (H_mat @ (Psi_i + k3      )).ravel()
        Psi_n = Psi_i + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0
        # Psi_n /= (np.linalg.norm(Psi_n))
        return Psi_n
    
    def propRK4_Numba(self): # Hs: 10*N*N, Psi_arr: 10*N #NUMBA
        k1 = (-1j) * self.dt * H_Psi_Numba(self.H_aux, (self.Psi_arr           ))
        k2 = (-1j) * self.dt * H_Psi_Numba(self.H_aux, (self.Psi_arr + 0.5 * k1))
        k3 = (-1j) * self.dt * H_Psi_Numba(self.H_aux, (self.Psi_arr + 0.5 * k2))
        k4 = (-1j) * self.dt * H_Psi_Numba(self.H_aux, (self.Psi_arr +       k3))
        self.Psi_arr += (k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0)
        # self.Psi_arr /= (np.linalg.norm(self.Psi_arr, axis=1).reshape(-1,1))
        return

    def propCN_Numba(self): # Hs: 10*N*N, Psi_arr: 10*N #NUMBA
        Psi_np1 = self.Psi_arr  
        Psi_np1_new = self.Psi_arr - 0.5 * 1j * self.dt * H_Psi_Numba(self.H_aux, (Psi_np1 + self.Psi_arr))
        while np.max(abs(Psi_np1_new - Psi_np1)) > 1e-6:
            Psi_np1 = Psi_np1_new
            Psi_np1_new = self.Psi_arr - 0.5 * 1j * self.dt * H_Psi_Numba(self.H_aux, (Psi_np1 + self.Psi_arr))
        self.Psi_arr = Psi_np1_new
        return        
    
    #def propCN_HW4(u0, t):
    def propCN_HW4(self):
        Mf = np.eye(self.H_size) + 0.5*self.dt/(1j)*self.H_aux # 半個 Forward Difference
        Mb = np.eye(self.H_size) - 0.5*self.dt/(1j)*self.H_aux # 半個 Backward Difference
        self.Psi_arr = np.linalg.solve(Mb, H_Psi_Numba(Mf, self.Psi_arr))
        return

    def evalF_Numba(self, Psi=None): #Psi_array: ik
        # Evaluate F only with current Psi_n
        # N_psi = self.aux_size # N_psi = 2 in 1D, 10 in 2D

        # If Psi is not provided: 
        if Psi is None:
            Psi = self.Psi_arr
                         
        # update expectation values 
        # x_tensor: mjk, ij, ik => ev_x_t/ev_p_t: im 
        ev_x_t = expectation_Numba(self.x_tensor, Psi).real
        ev_p_t = expectation_Numba(self.p_tensor, Psi).real
        
        # update F operators
        # rand_f_t = np.zeros(ev_p_t.shape) # same shape to ev_p_t        
        # rand_f1_t = np.random.normal(0, 2*eta1*(1.38e-23*Temperature/4.36e-18),(1,1))
        gamma_t = self.friction_coefficient * ev_p_t #- rand_f_t #i*m, scalar, i*m
        F = F_Numba(self.x_tensor, gamma_t, ev_x_t) # #mjk, im, im
        return F    
    
    def evalF_single(self, Psi_n):
        # Evaluate F only with current Psi_n
        return self.evalF_Numba(Psi_n.reshape(1,-1))

    def update_V(self, t_pulse):
        # +k pulse excite ket side
        # pulses: [[3000, 5, 0.005], ]
        delta_t = (t_pulse-self.t_vec)
        omega_vec = np.array([x[0] for x in self.pulses])
        tau_p = np.array([x[1] for x in self.pulses])
        I_pulse = np.array([x[2] for x in self.pulses])
        c_env = -4*np.log(2)/tau_p**2
        exp_term = I_pulse * np.exp(c_env*delta_t**2) * np.exp(1j*omega_vec*delta_t)
        self.V = self.X_mu.reshape(1, self.H_size, self.H_size) * exp_term.reshape(-1,1,1)
        return 

    def Initialize_Psi(self, Psi_0 = None, bool_orig2eigen=False):
        if Psi_0 is None:
            # initial Psi_0 on eigen
            Psi_0 = np.zeros(self.H_size, dtype=complex)
            Psi_0[0] = 1
        elif type(Psi_0) is int:
            # initial Psi_0 on orig
            Psi_ind = Psi_0
            Psi_0 = np.zeros(self.H_size, dtype=complex)
            Psi_0[Psi_ind] = 1
        elif type(Psi_0) is list:
            if len(Psi_0) != self.H_size:
                raise Exception("The length of initial Psi_0 %d \
                                 is not consistent to Hamiltonian size %d" \
                                 %(len(Psi_0), self.H_size))
            Psi_0 = np.array(Psi_0)
        Psi_0 /= np.linalg.norm(Psi_0) # normalize

        if bool_orig2eigen:
            Psi_0 = self.U.T@Psi_0

        self.Psi_arr = np.array([Psi_0]*self.aux_size)
        return
    
    # The following functions are just defined as interface;
    # They should be defined in the sub-classes.

    # Data Manipulation
    def Read_Matrices(self):
        if self.param["Huxp_source"] == "current_yml":
            if self.worker_id == 0:
                print("Trying to read H, mu, x and p from input yml file directly.")
            self.Read_Matrices_yml()
        elif type(self.param["Huxp_source"]) == str:
            if self.worker_id == 0:
                print("Trying to read H, mu, x and p from pickle/npz file: %s"%self.param["Huxp_source"])
            self.Read_Matrices_pickle(self.param["Huxp_source"])
        return

    def Read_Matrices_yml(self):
        def arr_helper(x):
            lst = self.param[x]
            res = np.array(lst)
            if res.dtype.kind in ["U", "S", "O"]:
                res = np.char.replace(res, "i", "j").astype(complex)
            return res
        self.read = [arr_helper(x) for x in ["H", "mu", "x", "p"]]
        self.H_size = self.read[0].shape[0]        
        self.N_mode = self.read[2].shape[0]        
        self.read = [np.eye(self.H_size)] + self.read   
        return
    
    def Read_Matrices_pickle(self, filename):
        data_input = np.load(filename, allow_pickle=True)
        H_read = data_input['H']
        mu_read = data_input['mu'] * DEBYE2AU # #Debye to a.u.        
        try: # format specified in tutorial
            x_read = data_input['x'] # mode*N*N
            p_read = data_input['p'] # mode*N*N
        except KeyError: # for QR's deprecated format
            select_mode = np.array([key for key in data_input if type(key)==int])                            
            x_read = np.array([data_input[key][0] for key in select_mode]) # mode*N*N
            p_read = np.array([data_input[key][1]-1j*data_input[key][2] for key in select_mode]) # mode*N*N

        self.H_size = H_read.shape[0]        
        self.N_mode = x_read.shape[0]
        self.read = [np.eye(self.H_size), H_read, mu_read, x_read, p_read]        
        return 
    
    def Transform_Matrices_into_eigenbasis(self):
        # Convert Operators into eigenbasis
        _, H_read, mu_read, x_read, p_read = self.read
        
        [E,Q] = np.linalg.eigh(H_read) # obtain eigenbasis         
        H_read = np.diag(E) # equivalent to Q.T @ H_read @ Q

        path_info = ['einsum_path', (0, 1), (0, 1)] # Precalculated einsum Path
        mu_read = np.einsum("ijk,jm,kn->imn", mu_read, Q, Q, optimize=path_info)         
        x_read  = np.einsum("ijk,jm,kn->imn", x_read,  Q, Q, optimize=path_info)
        p_read  = np.einsum("ijk,jm,kn->imn", p_read,  Q, Q, optimize=path_info)
        mu_read = np.linalg.norm(mu_read, axis=0)
        mu_read = np.triu(mu_read, 1) # ready for X_mu
        self.read = [Q, H_read, mu_read, x_read, p_read]
        # self.U = Q
        return

    def Population(self): # Didn't Change Basis...
        return (self.Psi_arr[0].conj() * self.Psi_arr[0]).real

    def update_H_aux_Numba(self):
        raise Exception("Need to define 'update_H_Numba(self)' before actual use.")

    def P_current(self):
        raise Exception("Need to define 'P_current(self)' before actual use.")

    def Initialize_t_vec(self, t_vec=None, omega_vec=None):
        raise Exception("Need to define 'Initialize_t_vec(self)' before actual use.")

