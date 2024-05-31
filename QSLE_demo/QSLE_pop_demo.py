import sys
import pickle
import numpy as np
from QSLE import QSLE_pop, read_input

if __name__ == "__main__": 
    input_yml = "qsle_1D_pop.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]

    param = read_input(input_yml)
    qsle_pop = QSLE_pop(param)
    qsle_pop.Preprocessor()
    qsle_pop.Execute()

    if qsle_pop.param["population_timestep"]:
        np.savez_compressed("qsle_pop.npz", t_lst=qsle_pop.t_lst, pop=qsle_pop.pop)

