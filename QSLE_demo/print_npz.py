import sys
import numpy as np

if __name__ == "__main__": 
    if len(sys.argv) > 1:
        input_npz = sys.argv[1]

    npzfile = np.load(input_npz)
    t_lst = npzfile["t_lst"].reshape(-1,1)
    for key in npzfile:
        if key in ["t_lst", "T_lst", "tau_lst"]: # skip them
            continue
        elif key == "P":
            P = npzfile[key].reshape(-1,1)
            key_mat = np.hstack((t_lst, P.real, P.imag))
            header = "t (fs), P.real, P.imag" 
        elif key == "pop":
            pop = npzfile[key]
            key_mat = np.hstack((t_lst, pop)) # pop is real
            header = "t (fs), " + ", ".join(("%d"%i for i in range(pop.shape[1])))
        np.savetxt(key + ".txt", key_mat, fmt="% 14.8f", 
                   delimiter=", ", header=header)
