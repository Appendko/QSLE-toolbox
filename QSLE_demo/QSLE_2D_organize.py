import sys
import numpy as np
from QSLE import read_input, organize_P, plot_2dir

if __name__ == "__main__": 
    input_yml = "qsle_2D.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]  
    param = read_input(input_yml)

    if param.get("detect_timestep", 1):
        bool_remove = not(param.get("DEBUG", False))
        # tau_lst, t_lst from object 0
        P, T_lst, tau_lst, t_lst = organize_P(bool_remove) 
        np.savez_compressed("qsle_2D_P.npz", P=P, 
                            T_lst=T_lst, tau_lst=tau_lst, t_lst=t_lst)
        for P_i, T_i in zip(P, T_lst):
            plot_2dir(P_i, T_i, param)
            