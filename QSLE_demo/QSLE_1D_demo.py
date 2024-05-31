import sys
import pickle
import numpy as np
from QSLE import QSLE_1D, read_input, plot_ftir

if __name__ == "__main__": 
    input_yml = "qsle_1D.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]

    param = read_input(input_yml)
    qsle_1D = QSLE_1D(param)
    qsle_1D.Preprocessor()
    qsle_1D.Execute()

    if qsle_1D.param["detect_timestep"]:
        np.savez_compressed("qsle_1D_P.npz", t_lst=qsle_1D.t_lst, P=qsle_1D.P)
        plot_ftir(qsle_1D.P, param)
            