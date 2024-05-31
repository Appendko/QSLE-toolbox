#!/bin/bash
#PBS -N IR_2D
#PBS -o IR_2D.out
#PBS -e IR_2D.err
#PBS -l nodes=6:ppn=16
#PBS -q short-e5

### Begin INPUT BLOCK ###
#INPUT=qsle_2D_model.yml
INPUT=qsle_2D.yml
codepath=/path_to_QSLE_toolbox/QSLE_demo
#### End INPUT BLOCK ####

echo "Starting on `hostname` at `date`"
source ~/.bashrc

cd $PBS_O_WORKDIR
if [ -n "$PBS_NODEFILE" ]; then
   if [ -f $PBS_NODEFILE ]; then
      echo "Nodes used for this job:"
      cat ${PBS_NODEFILE}|uniq
   fi
fi

mpirun -hostfile $PBS_NODEFILE --npernode 1 --bind-to none python $codepath/QSLE_2D_demo.py $INPUT
python $codepath/QSLE_2D_organize.py $INPUT

echo "Job Ended at `date`"
echo '======================================================='



