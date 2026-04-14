#!/bin/bash

#======================================================
#
# Job script for running a serial job on a single core 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=40
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=48:00:00
#
# Job name
#SBATCH --job-name=Assignment4LatticeSweep
#
# Output file
#SBATCH --output=Assignment4LatticeSweepOutput
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements

module load mpi/latest


#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program
cd /users/gmb22123/PH510/Assignment4/

# time python3 FixedCode.py This will only use one rank

#mpirun -np 1 python3 Assignment4Parallelism.py
#mpirun -np 2 python3 Assignment4Parallelism.py
#mpirun -np 4 python3 Assignment4Parallelism.py
#mpirun -np 8 python3 Assignment4Parallelism.py
#mpirun -np 16 python3 Assignment4Parallelism.py
#mpirun -np 32 python3 Assignment4Parallelism.py
mpirun -np 40 python3 assignment4.py







#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
