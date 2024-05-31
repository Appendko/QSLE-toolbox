import sys
import numpy as np
from QSLE import read_input, plot_2dir

if __name__ == "__main__": 
    input_yml = "qsle_2D.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]  
    param = read_input(input_yml)

    input_npz = "qsle_2D_P.npz"
    if len(sys.argv) > 2:
        input_npz = sys.argv[2]   
        
    if param.get("detect_timestep", 1):
        P_npz = np.load(input_npz)
        P, T_lst = [P_npz[x] for x in ["P", "T_lst"]]
        for P_i, T_i in zip(P, T_lst):
            plot_2dir(P_i, T_i, param)
