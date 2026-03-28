#!/usr/bin/env python3
# Â©(2026) ADAM HUSSAIN. ALL RIGHTS RESERVED
# Python Version: python3 --version: Python 3.9.21

"""
Simulation of the 2D Ising model using Metropolis Monte Carlo
""" 

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

#=====================================#
# Defining Lattice


L = 16   # Range of lattice sizes
H = 0              # External magnetic field set to 0
J = 1              # Ferromagnetic model

# The energy with H = 0 reduces to E = -J * sum(i,j) of s_i s_j
# J = 1 so this reduces further to E = -sum(i,j) of s_i s_j


def lattice():
    """
    Function which creates a lattice filled with
    randomly oriented spins +- 1
    """
    lattice = np.zeros((L,L), dtype=int)
    
    for i in range(L):      # Using two loops since we're in 2D
        for j in range(L):  # Using two loops since we're in 2D
            lattice[i,j] = random.choice([-1,1]) # Equal chance of spin being -1 or 1
            
    return lattice          # Returns a L x L grid of random spins
    

spins = lattice()
print(spins)                # Making sure it prints

def ising_energy(spins):   # Ising Energy which is the energy of a ferromagnetic system
                           # based on the spins of neighbouring particles is equal to the
                           # negative of the interaction strength multiplied by the sum of
                           # the product of spins of the neighburing atom
    """
    The ising energy of the 2D Lattice
    """
    energy = 0             # Energy initially zero, gets updated throughout the loop.
    
    for i in range(L):
        for j in range(L):
            spin = spins[i,j]  # Extracts each spin from the lattice
                               # The <i,j> in the formula means nearest neighours
