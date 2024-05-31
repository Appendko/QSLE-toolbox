import os
import itertools
# import cProfile
from multiprocessing import Pool, Process, Queue
import numpy as np
from QSLE.base_QSLE import base_QSLE
from QSLE.common import DEBYE_SI, DIP_AU_SI, DEBYE2AU, T_FS2AU, WN2AU
from QSLE.common import timestr, logtime, timer_func
from QSLE.common import serialize, deserialize

class QSLE_2D(base_QSLE):
    default_param = { 
            "molecule_system": "Molecule",
            "T_lst": [0, 100],
            "friction_coefficient": 0.005,   
            "detect_time_duration": 1000,
            "detect_timestep": 4,
            "pulses": [[3000, 5, 0.005], [3000, 5, 0.005], [3000, 5, 0.005]],
            "propagating_timestep":  0.25,
            "bool_tensor": False,
            "Nprocess": os.cpu_count(),
            "DEBUG": False,
            "bool_orig2eigen": False,
        }
        
    def __init__(self, param):
        super().__init__(self.default_param)                
        self.update_param(param)
        self.aux_size = 10 
        return

    def Serialize_Matrices(self):
        # H_read, mu_read, x_read, p_read = self.read
        def read_helper(read_i):
            return serialize(read_i, read_i.dtype), read_i.dtype, read_i.shape
        global arr_lst
        arr_lst = [read_helper(read_i) for read_i in self.read]
        self.read = None        
        return 
    
    def Deserialize_Matrices(self):
        global arr_lst
        self.U, self.H, self.X_mu, self.x_tensor, self.p_tensor = [deserialize(*arr_i) for arr_i in arr_lst]
        return

    def update_H_aux_Numba(self):
        # 2D-case: 10 aux Schrodinger Equations
        [V1_k, V2_b, V3_b] = self.V
        V1_b = V1_k.T.conj()
        V2_k = V2_b.T.conj()
        V3_k = V3_b.T.conj()

        self.H_aux = self.evalF_Numba()
        self.H_aux += self.H.reshape(1, self.H_size, self.H_size) # Use the same space for memory       
        self.H_aux[0] = self.H_aux[0]          
        self.H_aux[1] = self.H_aux[1] - V1_b
        self.H_aux[2] = self.H_aux[2] - V2_k
        self.H_aux[3] = self.H_aux[3] - V3_k
        self.H_aux[4] = self.H_aux[4] - V1_b - V2_b
        self.H_aux[5] = self.H_aux[5] - V1_b - V3_b
        self.H_aux[6] = self.H_aux[6] - V2_k - V3_k
        self.H_aux[7] = self.H_aux[7] - V1_k - V2_k
        self.H_aux[8] = self.H_aux[8] - V1_k - V3_k
        self.H_aux[9] = self.H_aux[9] - V1_k - V2_k - V3_k
        return

    def update_V_lambda(self, t_pulse):
        delta_t = (t_pulse-self.t_vec)
        exp_term = self.I_pulse * np.exp(self.c_env*delta_t**2) * np.exp(1j*self.omega_vec*delta_t)
        V1_k = lambda: self.X_mu * exp_term[0]
        V2_b = lambda: self.X_mu * exp_term[1]
        V3_b = lambda: self.X_mu * exp_term[2]
        self.V = [V1_k, V2_b, V3_b]
        return 

    def update_H_lambda(self):
        # 2D-case: 10 aux Schrodinger Equations
        # Psi_0, Psi_1, Psi_2, Psi_3, \
        # Psi_12, Psi_13, Psi_23, \
        # Psi_12_2, Psi_13_2, Psi_123 = self.Psi_arr       
        [V1_k, V2_b, V3_b] = self.V
        V1_b = lambda: V1_k().T.conj()
        V2_k = lambda: V2_b().T.conj()
        V3_k = lambda: V3_b().T.conj()

        H_0    = lambda Psi_i: self.H + self.evalF_single(Psi_i)         
        H_1    = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_b()
        H_2    = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V2_k()
        H_3    = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V3_k()
        H_12   = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_b() - V2_b()
        H_13   = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_b() - V3_b()
        H_23   = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V2_k() - V3_k()
        H_12_2 = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_k() - V2_k()
        H_13_2 = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_k() - V3_k()
        H_123  = lambda Psi_i: self.H + self.evalF_single(Psi_i) - V1_k() - V2_k() - V3_k()
        self.H_aux = [H_0, H_1, H_2, H_3, H_12, H_13, H_23, H_12_2, H_13_2, H_123]
        return 
    
    def P_current(self):
        Psi_n_0, Psi_n_1, Psi_n_2, Psi_n_3, \
        Psi_n_12, Psi_n_13, Psi_n_23, \
        Psi_n_12_2, Psi_n_13_2, Psi_n_123 = self.Psi_arr
        
        aux_b_1 = Psi_n_13.conj() - Psi_n_1.conj()
        aux_b_2 = Psi_n_12.conj() - Psi_n_1.conj()
        aux_b_3 = Psi_n_1.conj() - Psi_n_0.conj()
        aux_b_4 = Psi_n_0.conj()
        aux_k_1 = Psi_n_2 - Psi_n_0
        aux_k_2 = Psi_n_3 - Psi_n_0
        aux_k_3 = Psi_n_23 - Psi_n_2 - Psi_n_3 + Psi_n_0
        aux_k_4 = Psi_n_123 - Psi_n_12_2 - Psi_n_13_2 - Psi_n_23 + \
                Psi_n_2 + Psi_n_3
        
        P_se  = aux_b_1 @ self.X_mu @ aux_k_1
        P_gsb = aux_b_2 @ self.X_mu @ aux_k_2 + aux_b_4 @ self.X_mu @ aux_k_4
        P_esa = aux_b_3 @ self.X_mu @ aux_k_3

        P_t = P_se + P_gsb + P_esa
        return P_t

    def Initialize_t_vec(self, tau_input, T_input):
        # t_vec and omega_vec should be in AU
        if tau_input >= 0: #ordering: (tau_1, tau_2, tau_3)
            self.t_vec = self.param['t0'] + np.array([0, tau_input, tau_input+T_input], dtype=float)
        else: #ordering: (tau_2, tau_1, tau_3)
            self.t_vec = self.param['t0'] + np.array([-tau_input, 0, -tau_input+T_input], dtype=float)
        self.t_vec *= T_FS2AU    
        return 

    def propRK4(self):
        return self.propRK4_Numba()

    def update_H_aux(self): # just rename it to keep the interface consistent
        return self.update_H_aux_Numba()    

    def update_param(self, param):
        self.param.update(param)
        if self.param["Nprocess"] == 0:
            self.param["Nprocess"] = os.cpu_count()

        # t0
        self.param["t0"] = self.param["pulses"][0][1] * 4 

        # pulse
        self.pulses = self.param["pulses"]    
        for x in self.pulses: # omega, tau_p, I_pulse
            x[0] *= WN2AU # omega
            x[1] *= T_FS2AU # omega

        # update frequently-used parameters
        self.dt = self.param["propagating_timestep"] * T_FS2AU
        self.friction_coefficient = self.param["friction_coefficient"] / T_FS2AU 
        self.bool_tensor = self.param["bool_tensor"]
        return

    def t_lst_estimate(self):
        # Actual t_lst depends on tau, but it's weird to store different t_lst for different tau
        # Just estimate one when we collect data with Organize_P ...
        Npoint = self.param["detect_time_duration"] // self.param["detect_timestep"] + 1
        t_lst = np.linspace(0, self.param["detect_time_duration"], int(Npoint))
        return np.array(t_lst)

    def Signal_2DIR(self, tau_input, T_input):
        # pulse central times and simulation time
        self.Initialize_t_vec(tau_input, T_input)

        # define parameter: spectral initialization
        total_time = self.param["detect_time_duration"]
        total_time_au = total_time * T_FS2AU + self.t_vec[2]
        detect_timestep_au = self.param["detect_timestep"] * T_FS2AU
        time_steps = np.arange(0, total_time_au + 1e-2*self.dt, self.dt)
        max_detect_step = self.param["detect_time_duration"] // self.param["detect_timestep"] + 1

        P = []
        t_next_P = self.t_vec[2] # The next target time to record P. In au.     

        detect_count = 0 # this should be worse...
        for i_t, t_pulse in enumerate(time_steps): # t_pulse is in au
            #print(i_t,"/", time_steps.shape[0], t_pulse) 
            if self.param["detect_timestep"] and t_pulse >= t_next_P: 
                P.append(self.P_current())
                t_next_P += detect_timestep_au # this shoule be better
                # detect_count += 1
                # t_next_P = self.t_vec[2] + detect_count * detect_timestep_au  # this should be worse...

            if self.bool_tensor == True:
                self.update_V(t_pulse)
                self.update_H_aux()            
                self.propRK4()  
            else: 
                self.update_V_lambda(t_pulse)            
                self.update_H_lambda()       
                Psi_next = []
                for Psi_i, H_lambda in zip(self.Psi_arr, self.H_aux):
                    Psi_next.append(self.propRK4_lambda(Psi_i , H_lambda))
                self.Psi_arr = np.array(Psi_next)
        if len(P) < max_detect_step:
            P += [0] * (max_detect_step-len(P))
        return np.array(P)
         
    def Preprocessor(self):
        os.environ['MKL_NUM_THREADS'] = "%d"%os.cpu_count()
        os.environ['OMP_NUM_THREADS'] = "%d"%os.cpu_count()
        self.Read_Matrices()
        self.Transform_Matrices_into_eigenbasis()
        self.Serialize_Matrices()
        return
        
    def Worker(self, tau_input, T_input):
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        self.Deserialize_Matrices()

        self.Initialize_Psi()
        P = self.Signal_2DIR(tau_input, T_input)
        return tau_input, T_input, P

    @timer_func    
    def Execute(self):
        # Use OpenMPI to Parallelize
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:        
            self.worker_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.worker_id = int(os.environ['OMPI_COMM_WORLD_RANK'])           
            if self.worker_id == 0:
                print("OpenMPI environment Detected.")              
        else: # Single Node
            self.worker_size, self.worker_id = 1, 0
        if self.worker_id == 0:
            print("Totally %d Nodes are used in this calculation."%(self.worker_size))

        self.T_lst = self.param["T_lst"]
        t = self.param["detect_time_duration"]
        step_size = self.param["detect_timestep"]
        self.tau_lst = np.arange(-t, t + 1e-5*step_size, step_size)
        TAU_T_iterators = list(itertools.product(self.tau_lst, self.T_lst))[self.worker_id::self.worker_size]

        pool = Pool(self.param["Nprocess"])
        res_lst = pool.starmap(self.Worker, TAU_T_iterators)        
        res_lst = np.array(res_lst, dtype=object)
        self.t_lst = self.t_lst_estimate()
        np.savez_compressed("qsle_2D_%d.npz"%self.worker_id, res=res_lst, 
                            worker_size=self.worker_size, tau_lst=self.tau_lst, t_lst=self.t_lst)
        return

def organize_P(bool_remove = False):
    import itertools as it
    tau_lst, T_lst, P_lst = [], [], []

    qsle_0 = np.load("qsle_2D_0.npz")
    worker_size, tau_lst_0, t_lst_0 = [qsle_0[x] for x in ["worker_size", "tau_lst", "t_lst"]]
    print("worker_size:", worker_size)

    res_lst = []
    for i in range(worker_size):
        res_npz = np.load("qsle_2D_%d.npz"%i, allow_pickle=True)
        res = res_npz["res"]
        res_lst.append(res)

    res_lst = (x for x in it.chain(*(it.zip_longest(*res_lst))) if x != None)
    tau_lst, T_lst, P_lst = list(zip(*res_lst))
    tau_lst = np.hstack(tau_lst)
    T_lst = np.hstack(T_lst)
    P_lst = np.vstack(P_lst)    

    # reshape for easier indexing
    T_lst_uniq = np.unique(T_lst)
    tau_T_shape = np.unique(tau_lst).shape[0], T_lst_uniq.shape[0]
    P = P_lst.reshape(*tau_T_shape, -1).transpose([1,0,2])

    if bool_remove:
        for i in range(worker_size):
            os.remove('qsle_2D_%d.npz'%i)

    return P, T_lst_uniq, tau_lst_0, t_lst_0

def plot_2dir(P, T, param):
    # Import
    import matplotlib
    try:
        ipython_env = get_ipython().__class__.__name__
        print("IPython detected, using TkAgg Backend in matplotlib.")
        matplotlib.use('TkAgg')
        bool_ipython = True
    except NameError: # regular python: execute from command line
        print("regular Python detected, using Agg Backend in matplotlib.")
        matplotlib.use('Agg')
        bool_ipython = False

    import matplotlib.pyplot as plt
    from scipy.signal import hann
    from scipy.fft import fft2, fftshift, fftfreq, next_fast_len
    from scipy.constants import physical_constants as phys_const

    def x_helper(L_t, step, xrange, power):
        n_t = next_fast_len(L_t*2**power)
        c = phys_const['speed of light in vacuum'][0] * 100 * 1e-15 # in cm/fs    
        x_plot = fftshift(fftfreq(n_t, step))/c
        t_min = np.flatnonzero(-x_plot < xrange[1])[0]
        t_max = np.flatnonzero(x_plot > -xrange[0])[0]
        slice_t = slice(t_min, t_max)
        if t_max < t_min:
            slice_t = slice(t_min, t_max, -1)
        return n_t, x_plot[slice_t], slice_t

    spec_range = param["spec_range"]
    t_step_t = param["detect_timestep"]
    t_step_tau = param["detect_timestep"]
    freq_range_t = np.array(spec_range)
    freq_range_tau = np.array(spec_range)

    L_tau = (P.shape[0]+1)//2
    mid_tau = P.shape[0]//2 
    L_t = P.shape[1]

    win_t = hann(L_t*2)[L_t:].reshape(1, -1)
    win_tau = hann(L_tau*2)[L_tau:].reshape(-1, 1)

    PP_r = P[mid_tau:].T 
    PP_n = P[mid_tau::-1].T #negative tau
    PP_h_r = PP_r * win_t * win_tau
    PP_h_n = PP_n * win_t * win_tau

    freq_range_tau_n = freq_range_tau
    freq_range_tau_r = -freq_range_tau_n[::-1]

    n_t, t_plot, slice_t = x_helper(L_t, t_step_t, freq_range_t, 3)
    n_tau, tau_plot_r, slice_tau_r = x_helper(L_tau, t_step_tau, freq_range_tau_r, 5)
    n_tau, tau_plot_n, slice_tau_n = x_helper(L_tau, t_step_tau, freq_range_tau_n, 5)

    y_r = fftshift(fft2(1j * PP_h_r, (n_t, n_tau))) /n_t/n_tau 
    y_r_p = y_r[slice_t][:,slice_tau_r]
    # y_r_plot = y_r_p.real / np.abs(y_r.real).max()        
    
    y_n = fftshift(fft2(1j * PP_h_n, (n_t, n_tau))) /n_t/n_tau 
    y_n_p = y_n[slice_t][:,slice_tau_n]
    # y_n_plot = y_n_p.real / np.abs(y_n.real).max()    

    y = y_r + np.fliplr(np.roll(y_n,-1,axis=1))
    y_p = y[slice_t][:,slice_tau_r] 
    y_plot = y_p.real / np.abs(y_p.real).max()   

    # Plotting
    if bool_ipython:
        plt.ion()

    plt.figure()
    plt.contourf(tau_plot_r, -t_plot, y_plot, np.arange(-1,1+1e-15,0.01), cmap="seismic", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlim(spec_range)
    plt.ylim(spec_range)
    plt.title("T=%d, y_r"%T)
    plt.xlabel(r'$\omega_\tau$ (cm$^{-1}$)')
    plt.ylabel(r'$\omega_\t$ (cm$^{-1}$)')    
    plt.savefig('2DIR_T_%d.png'%T, bbox_inches='tight')   
    return plt.gcf(), plt.gca()
