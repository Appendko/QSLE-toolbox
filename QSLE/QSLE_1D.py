import numpy as np
from QSLE.base_QSLE import base_QSLE
from QSLE.common import DEBYE_SI, DIP_AU_SI, DEBYE2AU, T_FS2AU, WN2AU
from QSLE.common import timestr, logtime, timer_func

class QSLE_1D(base_QSLE):
    default_param = {
            "molecule_system": "Molecule",
            "friction_coefficient": 0.005,
            "pulses": [[3000, 5, 0.005], ],            
            "propagating_timestep": 0.25,            
            "detect_time_duration": 8001,
            "detect_timestep": 4,
            "Psi_0": None,
            "bool_orig2eigen": False,
        }
        
    def __init__(self, param):
        super().__init__(self.default_param)        
        self.update_param(param)
        self.aux_size = 2 
        return   

    def update_H_aux_Numba(self):
        # 1D-case: 2 aux Schrodinger Equations
        [V1_k] = self.V
        V1_b = V1_k.T.conj()

        self.H_aux = self.evalF_Numba()
        self.H_aux += self.H.reshape(1, self.H_size, self.H_size) # Use the same space for memory
        
        self.H_aux[0] = self.H_aux[0]         # H_1 = H + F_1
        self.H_aux[1] = self.H_aux[1] - V1_b  # H_0 = H + F_0  - V1_k        
        return

    def P_current(self):
        Psi_n_0, Psi_n_1 = self.Psi_arr
        aux_b_0 = Psi_n_0.conj()
        aux_k_0 = Psi_n_1 - Psi_n_0
        P_t = aux_b_0 @ self.X_mu @ aux_k_0
        return P_t

    def Initialize_t_vec(self, t_vec=None):       
        # t_vec and omega_vec should be in AU
        # Default: read "t0" and "wn_omega" in param to generate them        
        if t_vec is None:
            tau_1 = self.param['t0']
            t_vec = np.array([tau_1] ,dtype=float)
            t_vec *= T_FS2AU
           
        self.t_vec = t_vec
        return 

    def propRK4(self):
        return self.propRK4_Numba()

    def update_H_aux(self): # just rename it to keep the interface consistent
        return self.update_H_aux_Numba()    

    def update_param(self, param):
        self.param.update(param)

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
        return

    def t_lst_estimate(self):
        # In principle 1D Jobs should use the Actual timestep, not this one.
        Npoint = self.param["detect_time_duration"] // self.param["detect_timestep"] + 1
        t_lst = np.linspace(0, self.param["detect_time_duration"], int(Npoint))
        return np.array(t_lst)

    def FTIR_signal(self):
        # pulse central times and simulation time
        self.Initialize_t_vec()

        # define parameter: spectral initialization
        total_time = self.param["t0"] + self.param["detect_time_duration"]
        total_time_au = total_time * T_FS2AU    
        detect_timestep_au = self.param["detect_timestep"] * T_FS2AU
        time_steps = np.arange(0, total_time_au, self.dt)

        t_lst = []
        P = []
        t_next_P = self.t_vec[0] # The next target time to record P. In au.

        for i_t, t_pulse in enumerate(time_steps): # t_pulse is in au
            #print(i_t,"/", time_steps.shape[0], t_pulse) 
            if self.param["detect_timestep"] and t_pulse >= t_next_P: 
                t_lst.append(t_pulse/T_FS2AU)                
                P.append(self.P_current())
                t_next_P += detect_timestep_au

            self.update_V(t_pulse)
            self.update_H_aux()            
            self.propRK4()  

        return np.array(t_lst), np.array(P)
         
    @timer_func
    def Preprocessor(self):
        self.Read_Matrices()
        self.Transform_Matrices_into_eigenbasis()
        return
    
    @timer_func
    def Execute(self):
        self.Initialize_Psi(self.param["Psi_0"])
        # Single process only, No need to deserialize, just extract from self.read
        self.U, self.H, self.X_mu, self.x_tensor, self.p_tensor = self.read
        self.read = None
        self.t_lst, self.P = self.FTIR_signal()
        return

def plot_ftir(P, param):
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
    from scipy.fft import fft, fftshift, fftfreq, next_fast_len
    from scipy.constants import physical_constants as phys_const

    # Spectrum Calculation
    spec_range = param["spec_range"]

    L_t = len(P) # Sampling length
    PP_h = P * hann(L_t*2)[L_t:]
    
    # Sampling frequency
    n_t = next_fast_len(L_t*2**4)
    
    # Frequency domain x axis
    t_step_t = param.get("detect_timestep", 4)
    c = phys_const['speed of light in vacuum'][0] * 100 * 1e-15 # in cm/fs
    x_plot = -fftshift(fftfreq(n_t, t_step_t))/c
    
    # Fourier transform
    # y = fftshift(fft(PP_h, n_t)) / n_t
    # y_plot = y.imag / np.abs(y.imag).max()
    y = fftshift(fft(1j*PP_h, n_t)/ n_t)
    y_plot = - y.real / np.abs(y.real).max()

    # Plotting
    if bool_ipython:
        plt.ion()
    plt.plot(x_plot, y_plot, linewidth=1.5)
    plt.gca().set_yticklabels([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlim(spec_range)
    plt.ylim([-0.05, 1])
    plt.xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=14)
    plt.savefig('FTIR.png', bbox_inches='tight')
    return plt.gcf(), plt.gca()
    
            
