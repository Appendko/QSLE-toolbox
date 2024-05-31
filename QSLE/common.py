import time
import yaml
import numpy as np
from multiprocessing import RawArray
from scipy.constants import physical_constants as phys_const

# Define Constants
DEBYE_SI = 1e-21/phys_const["speed of light in vacuum"][0]
DIP_AU_SI = phys_const["atomic unit of electric dipole moment"][0]
DEBYE2AU = 0.393456 # DEBYE_SI/DIP_AU_SI # 0.39343029437118193
T_FS2AU = 1e-15 / phys_const['atomic unit of time'][0] # 41.34137333518211
WN2AU = 1e2/phys_const['hartree-inverse meter relationship'][0] # 4.556335252911937e-06

# Timer Decorator
def timestr():
    return '['+time.asctime(time.localtime(time.time()))+'] '

def logtime(text):
    print(timestr()+text, flush=True)

def timer_func(func):
    def ret_func(*args, **kwargs):
        tic = time.time()
        func(*args, **kwargs)
        toc = time.time()
        logtime("%s: %f Seconds"%(func.__name__, toc-tic))
        return
    return ret_func

# Serialize / Deserialize for Share-memory RawArrays
def serialize(M_read, dtype):
    arr = RawArray('c', M_read.nbytes)
    M = np.frombuffer(arr, dtype=dtype)
    M[:] = M_read.ravel()
    del M
    return arr

def deserialize(arr, dtype, shape):
    return np.frombuffer(arr, dtype=dtype).reshape(shape)

def read_input(input_yml):
    with open(input_yml, "r") as f:
        param = yaml.load(f, Loader=yaml.SafeLoader)
    return param

# def expection_value_single(Operator, Psi_i):
#     denominator = Psi_i.conj()@Psi_i
#     expection_value_lst = [Psi_i.conj()@Operator_i@Psi_i for Operator_i in Operator]
#     return np.array(expection_value_lst)/denominator
