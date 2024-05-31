#!/bin/bash
#SBATCH -J IR_2D              # Job name
#SBATCH -o %x.out             # Path to the standard output file
#SBATCH -e %x.err             # Path to the standard error ouput file
#SBATCH --ntasks-per-node=56  # Number of cores per Node
#SBATCH -c 1                  # Number of cores per MPI task
#SBATCH -N 2                  # Number of nodes to be allocated
#SBATCH -p ctest              # Partiotion(Queue) name

### Begin INPUT BLOCK ###
INPUT=qsle_2D_model.yml
#INPUT=qsle_2D.yml
codepath=/path_to_QSLE_toolbox/QSLE_demo
#### End INPUT BLOCK ####

# BEGIN Script
echo "Starting on `hostname` at `date`"

# Modules
module purge
module load old-module
module load gcc/13.2.0
module load openmpi/4.1.6
module load pkg/Anaconda3

MPIEXE=mpirun

cd $SLURM_SUBMIT_DIR
echo Working directory is $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
echo SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE

# Actual Job
$MPIEXE --npernode 1 --bind-to none python $codepath/QSLE_2D_demo.py $INPUT
python $codepath/QSLE_2D_organize.py $INPUT

