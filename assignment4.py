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
N_SIMS = 1                   # Number of sweeps of the metropolis
N_SPLIT = N_SIMS // nproc # Splitting work among processors
temperatures = np.linspace(1, 10, 100, 10000, 100000) #Array of temperatures 
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
            spin = spins[i,j]                                           # Extracts each spin from the lattice
                                                                        # The <i,j> in the formula means nearest neighours
            r_neighbour = spins[i, (j+1) % L]                                     # (i,j) is the spin at location (i,j) which is
            d_neighbour = spins[(i+1) % L, j]                                     # multiplied by a neighbouring spin such that
            
            energy += -spin * (r_neighbour + d_neighbour)
    return energy                                                            # E = -(spin * neighbour spin) (summed over all (i,j))
    
E = ising_energy(spins)
print(E)                                                                    # So confusingly, s_i is the spin at (i,j)
                                                                       # and s_j is the spin at (i,j+1)

                                          # a b c d
                                          # e f g h          where (i,j) = a  s_i = a, s_j = (b, right neighbour) and (e, down neighbour)
                                          # i j k l          so the ising energy sums those: a*b + a*e for that location (i,j)
                                          

#For the metropolis algorithm, we're now interested in four neighbours per point.
def metropolis(spins, T):
    """
    One sweep of the lattice
    Using equation 9.15 from notes
    """
    for _ in range(L*L):                #One full sweep is L*L
        i = random.randint(0, L-1)      # Picking random location (i,j) to flip
        j = random.randint(0, L-1)

#Taking the four neighbours
        neighbours_sum = (
            spins[(i-1) % L, j] + #Down
            spins[(i+1) % L, j] + #Up
            spins[i, (j+1) % L] + #Right
            spins[i, (j-1) % L]   #Left
        )
        delta_E = 2*J*spins[i,j] * neighbours_sum  # Equation 9.15 fledged out
        
        #Acceptance or Rejection
        if delta_E <= 0:
            spins[i,j] = -1 * spins[i,j] # Flip the sign
        else:
            w = np.exp(-delta_E / T)   #If E > 0, this is the probability of acceptance
            if random.random() < w:
                spins[i,j] = -1 * spins[i,j] # Flip the sign
                
def ising_sim(T):
    """
    Sweeps of the ising model at a temperature T
    """
    spins = lattice()
    
    for _ in range():
        energies = []
        magnetisation = []
        metropolis(spins, T)
        energy = ising_energy(spins)
        M = np.sum(spins) #Magnetisation is the sum of all spins in the lattice.
        
        energies.append(E)      #Storage of the energy
        magnetisation.append(E) #Storage of the magnetisation
        
        
    return energies, magnetisation
        

if rank < nproc -1:
    start = rank * N_SPLIT
    end = start + N_SPLIT
else:
    start = rank * N_SPLIT
    end = N_SIMS
    
for T in temperatures:
    localE = 0
    localM = 0
    
    for _ in range(start,end):
        TotalE, TotalM = ising_sim(T)
        localE += TotalE
        localM += TotalM
        
    globalE = comm.reduce(localE, op=MPI.SUM, root=0)
    globalM = comm.reduce(localM, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"T={T}, E={globalE}, M={globalM}")
