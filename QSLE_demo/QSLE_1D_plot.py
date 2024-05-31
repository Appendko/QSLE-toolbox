import sys
import pickle
import numpy as np
from QSLE import read_input, plot_ftir

if __name__ == "__main__": 
    input_yml = "qsle_1D.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]
    param = read_input(input_yml)

    input_npz = "qsle_1D_P.npz"
    if len(sys.argv) > 2:
        input_npz = sys.argv[2]   
        
    if param.get("detect_timestep", 1):
        P_npz = np.load(input_npz)
        P = P_npz["P"]
        plot_ftir(P, param)
