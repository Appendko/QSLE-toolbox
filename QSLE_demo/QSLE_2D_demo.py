import sys
import pickle
import numpy as np
from QSLE import QSLE_2D, read_input

if __name__ == "__main__": 
    input_yml = "qsle_2D.yml"
    if len(sys.argv) > 1:
        input_yml = sys.argv[1]
   
    param = read_input(input_yml)
    qsle_2D = QSLE_2D(param)
    qsle_2D.Preprocessor()
    qsle_2D.Execute()
