# QSLE-toolbox 

Ref: [J. Chem. Phys. 153, 154107 (2021)](https://doi.org/10.1063/5.0042848)

## Install
  - install: `pip install -e .` 
    - If you need to use it with a batch script, please make sure you installed QSLE-toolbox in the python environment used in the script.
  - uninstall: `pip uninstall QSLE`

## 1D FTIR Simulation on 2-mode model

### Model setup  
- Here we use `QSLE_demo/qsle_1D_model.yml` as the input file.   
  - The keyword `Huxp_source: current_yml` means we describe the necessary operators in this yml directly.
  - The system is a Two mode (1600 cm-1, 3200 cm-1) model system with 50 cm-1 Fermi-Resonance Coupling:
    - please check `QSLE_demo/model.ipynb` for the following operators:    
      - H: Hamiltonian, N-by-N matrix
      - mu: dipole moment operator, (A, N, N) matrices
        - A is dimension of dipole moment, in most cases it should be 3.
      - x: position operators (x): (M, N, N) matrices
        - M is the number of vibrational modes.
        - written in mass-weighted coordinates
      - p: momentum operator (p): (M, N, N) matrices
        - written in mass-weighted coordinates
      - All of these matrices are written in atomic unit
  - These operator are copied into `QSLE_demo/qsle_1D_model.yml`

### 1D FTIR Simulation
To perform **FTIR simulation**, we can use `QSLE_demo\QSLE_1D_demo.py`:

```python
python QSLE_1D_demo.py qsle_1D_model.yml
```

This code store the calculated Molecular Polarization in `qsle_1D_P.npz`, then perform FFT to obtain spectrum directly as `FTIR.png`. To print the contents in output `qsle_1D_P.npz` file, we can use `QSLE_demo\print_npz.py`:

```python
python print_npz.py qsle_1D_P.npz
```

To make the plots from the Molecular Polarization in `qsle_1D_P.npz`, we can use `QSLE_demo\QSLE_1D_plot.py`:

```
python QSLE_1D_plot.py qsle_1D_model.yml qsle_1D_P.npz
```

### Population Dynamics

To perform Population Dynamics simulation on this model system, we can use the same `qsle_1D_model.yml` file to describe the system. However, We need to specify Initial Wavefunction (Psi_0) in the yml file.
Uncomment the `# Psi_0: 3` to specify the initial wavefunction. In this case, 3 means the third basis (here we start from 0). Remember We only use this for Population Dynamics simulation; when you need to perform FTIR simulation, please comment out this line.
```bash
### Initial Wavefunction
# Psi_0: (wavefunction, integer for specific basis, or a full vector)
# Default Value: None
#     This should only uncommented for population calculation unless you really know the meaning.
Psi_0: 3
```

After uncommenting the `Psi_0` parameter, we can perform Population Dynamics on this system using `QSLE_demo\QSLE_pop_demo.py`:

```python
python QSLE_pop_demo.py qsle_1D_model.yml
```

The calculated population is then stored in `qsle_pop.npz`. To print the contents in this file, we can use the tool `QSLE_demo\print_npz.py` again: 

```python
python print_npz.py qsle_pop.npz
```


## 2D FTIR Simulation on Ar-Solvated Hydronium (H3O+Ar3)
### Model setup  
- Here we use `QSLE_demo/qsle_2D.yml` as the input file.   
  - The keyword `Huxp_source: H3O+Ar3.b+s.pickle` means we load the necessary operators from this existing file.
    - The system is H3O+Ar3 system considering 2 bending modes and 3 stretch modes. 
    - The input is should be a Python Pickle file Containing a Python Dictionary.
      - An '.npz' file from numpy is also valid.
    - It should contain the following keyword: "H", "mu", "x", "p", which corresponds to the 4 operators needed.

### 2DIR Simulation
To perform **2DIR simulation**, we can use `QSLE_demo\QSLE_2D_demo.py`. However, this require us to run a lot of different tau, t and T, which we can parallelize this calculatiopn to save time. Here we provide two example submission scripts, and please make necessary changes for the scripts:

- `IR_2D_pbs.sh` for PBS environment
- `IR_2D_slurm.sh`submission script for SLURM environment

By default we use OpenMPI to launch this program, so please ensure your mpirun is set to OpenMPI. Any version of OpenMPI should work. Meanwhile, please make sure you have installed QSLE-toolbox in the python environment used in the script.

```bash
qsub IR_2D_pbs.sh
```

The calculated Molecular Polarization is stored in `qsle_2D_P.npz`, and it automatically produces the 2D spectra as `2DIR_T_*.png`. To make the plots from the Molecular Polarization in `qsle_2D_P.npz`, we can use `QSLE_demo\QSLE_2D_plot.py`:

```
python QSLE_2D_plot.py qsle_2D.yml qsle_2D_P.npz
```

### Population Dynamics

To perform Population Dynamics simulation on this system, we can use the same `QSLE_pop_demo.py` tool. However, We need to specify Initial Wavefunction (Psi_0) in the yml file.
Uncomment the `# Psi_0: 3` to specify the initial wavefunction. In this case, 3 means the third basis (here we start from 0). Remember We only use this for Population Dynamics simulation; when you need to perform 2DIR simulation, please comment out this line.
```bash
### Initial Wavefunction
# Psi_0: (wavefunction, integer for specific basis, or a full vector)
# Default Value: None
#     This should only uncommented for population calculation unless you really know the meaning.
Psi_0: 3
```

After uncommenting the `Psi_0` parameter, we can perform Population Dynamics on this system using `QSLE_demo\QSLE_pop_demo.py`:

```python
python QSLE_pop_demo.py qsle_2D.yml
```

The calculated population is then stored in `qsle_pop.npz`. To print the contents in this file, we can use the tool `QSLE_demo\print_npz.py` again: 

```python
python print_npz.py qsle_pop.npz
```

