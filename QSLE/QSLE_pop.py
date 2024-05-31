import numpy as np
from QSLE.base_QSLE import base_QSLE
from QSLE.common import DEBYE_SI, DIP_AU_SI, DEBYE2AU, T_FS2AU, WN2AU
from QSLE.common import timestr, logtime, timer_func

class QSLE_pop(base_QSLE):
    default_param = {
            "molecule_system": "Molecule",
            "friction_coefficient": 0.005,
            "propagating_timestep": 0.25,            
            "detect_time_duration": 8001,
            "population_timestep": 1,
            "Psi_0": None,
            "bool_orig2eigen": True,
        }
        
    def __init__(self, param):
        super().__init__(self.default_param)        
        self.update_param(param)
        self.aux_size = 1 # Pop only, Only original H+F is left
        return   

    def update_H_aux_Numba(self):
        # Pop only, no aux Schrodinger Equations
        # Only original H+F is left
        self.H_aux = self.evalF_Numba()
        self.H_aux += self.H.reshape(1, self.H_size, self.H_size) # Use the same space for memory
        return

    def Population(self):        
        if self.param["bool_orig2eigen"]:
            psi = self.U.T @ self.Psi_arr[0]
        else: 
            psi = self.Psi_arr[0]
        return (psi.conj() * psi).real

    def propRK4(self):
        return self.propRK4_Numba()

    def update_H_aux(self): # just rename it to keep the interface consistent
        return self.update_H_aux_Numba()    

    def update_param(self, param):
        self.param.update(param)

        # update frequently-used parameters
        self.dt = self.param["propagating_timestep"] * T_FS2AU
        self.friction_coefficient = self.param["friction_coefficient"] / T_FS2AU 
        return

    def t_lst_estimate(self): 
        # In principle Population Jobs should use the Actual timestep, not this one.
        Npoint = self.param["detect_time_duration"] // self.param["population_timestep"] + 1
        t_lst = np.linspace(0, self.param["detect_time_duration"], int(Npoint))
        return np.array(t_lst)

    def Pop_Dynamics(self):
        # pulse central times and simulation time
        # self.Initialize_t_vec()

        # define parameter: spectral initialization
        total_time = self.param["detect_time_duration"]
        total_time_au = total_time * T_FS2AU 
        population_timestep_au = self.param["population_timestep"] * T_FS2AU
        time_steps = np.arange(0, total_time_au, self.dt)

        t_lst = []
        pop = []        
        t_next_pop = 0

        for i_t, t_pulse in enumerate(time_steps): # t_pulse is in au
            #print(i_t,"/", time_steps.shape[0], t_pulse) 
            if self.param["population_timestep"] and t_pulse >= t_next_pop: 
                t_lst.append(t_pulse/T_FS2AU)
                pop.append(self.Population())
                t_next_pop += population_timestep_au

            self.update_H_aux()            
            self.propRK4()  
            # self.propCN_Numba() # Debug Use: Convergence Test for RK4
            # self.propCN_HW4()   # Debug Use: Convergence Test for RK4
        return np.array(t_lst), np.array(pop)
         
    @timer_func
    def Preprocessor(self):
        self.Read_Matrices()
        self.Transform_Matrices_into_eigenbasis()
        return
    
    @timer_func
    def Execute(self):
        # Single process only, No need to deserialize, just extract from self.read
        self.U, self.H, self.X_mu, self.x_tensor, self.p_tensor = self.read
        self.read = None

        # Population Dynamics: 
        self.Initialize_Psi(self.param["Psi_0"], self.param["bool_orig2eigen"])
        self.t_lst, self.pop = self.Pop_Dynamics()
        return
