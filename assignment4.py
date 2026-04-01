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
N_SWEEPS = 1000                   # Number of sweeps of the metropolis
N_SPLIT = N_SWEEPS // nproc # Splitting work among processors
temperatures = np.linspace(1, 10, 30) #Array of temperatures 
delta_angle = np.pi
#=====================================#
# Defining Lattice


H = 0              # External magnetic field set to 0
J = 1              # Ferromagnetic model

# The energy with H = 0 reduces to E = -J * sum(i,j) of s_i s_j
# J = 1 so this reduces further to E = -sum(i,j) of s_i s_j


def lattice(L):
    """
    Function which creates a lattice filled with
    randomly oriented spins +- 1
    """
    grid = np.zeros((L,L), dtype=int)
    
    for i in range(L):      # Using two loops since we're in 2D
        for j in range(L):  # Using two loops since we're in 2D
            grid[i,j] = random.choice([-1,1]) # Equal chance of spin being -1 or 1
            
    return grid          # Returns a L x L grid of random spins

def ising_energy(spins, L):   # Ising Energy which is the energy of a ferromagnetic system
                              # based on the spins of neighbouring particles is equal to the
                              # negative of the interaction strength multiplied by the sum of
                              # the product of spins of the neighburing atom
    """
    The ising energy of the 2D Lattice
    """
    energy = 0             # Energy initially zero, gets updated throughout the loop.
    
    for i in range(L):
        for j in range(L):
                                                       # Extracts each spin from the lattice
                                                       # The <i,j> in the formula means nearest neighours
            r_neighbour = spins[i, (j+1) % L]                                     # (i,j) is the spin at location (i,j) which is
            d_neighbour = spins[(i+1) % L, j]                                     # multiplied by a neighbouring spin such that

            energy += -spins[i,j] * (r_neighbour + d_neighbour)
    return energy                                                            # E = -(spin * neighbour spin) (summed over all (i,j))
                                                                    # So confusingly, s_i is the spin at (i,j)
                                                                       # and s_j is the spin at (i,j+1)

                                          # a b c d
                                          # e f g h          where (i,j) = a  s_i = a, s_j = (b, right neighbour) and (e, down neighbour)
                                          # i j k l          so the ising energy sums those: a*b + a*e for that location (i,j)
                                          

#For the metropolis algorithm, we're now interested in four neighbours per point.
def metropolis(spins, T, L):
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
        delta_E = 2 * J * spins[i,j] * neighbours_sum  # Equation 9.15 fledged out
        
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
 
    energies = []
    magnetisation = []
  
    for _ in range(N_SWEEPS):
        metropolis(spins, T)
        energy = ising_energy(spins)
        M = np.sum(spins) #Magnetisation is the sum of all spins in the lattice.
        
        energies.append(energy)      #Storage of the energy
        magnetisation.append(M) #Storage of the magnetisation
        
        
    return np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2) # E, M and E^2
    # np.mean(energies**2) gives an error for some reason, np.array fixes it.  
    # These gets caught by localE, localM and localE2
        

if rank < nproc -1:
    start = rank * N_SPLIT
    end = start + N_SPLIT
else:
    start = rank * N_SPLIT
    end = N_SWEEPS
    
for T in temperatures:
    localE = 0
    localM = 0
    localE2 = 0
    
    for _ in range(start,end):
        TotalE, TotalM, TotalE2 = ising_sim(T)
        localE += TotalE
        localM += TotalM
        localE2 += TotalE2
        
    globalE = comm.reduce(localE, op=MPI.SUM, root=0)
    globalM = comm.reduce(localM, op=MPI.SUM, root=0)
    globalE2 = comm.reduce(localE2, op=MPI.SUM, root=0)
    
    if rank == 0:
        Cv = 1/1.38E-23 * T**2 * (globalE2 - E**2)
        print(f"T={T:.2f}, E={globalE:.2f}, M={globalM:.2f}, E2={globalE2:.2f}")
        
comm.Barrier()
if rank == 0:
    endtime = time.time()
    print(f"Number of processors: {nproc}")
    print(f"Timing: {endtime - starttime:.4f} seconds")
    
    plt.plot(temperatures, E)
    plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
    plt.close()
    
    plt.plot(temperatures, M)
    plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
    plt.close()
    
    plt.plot(temperatures, Cv)
    plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
    plt.close()
    

# Task 4 XY Model (Angle Incorporation)
# The X-Y model incorporates angles into the spins, so instead of
# spins up or down, they can have an infinite number of variations
# in between 0 and 2pi

def XY_lattice():
    """
    Function which creates a lattice filled
    with randomly oriented spins which can
    range from 0 - 2pi
    """
    
    lattice = np.zeros((L,L), dtype=float)
    
    for i in range(L):
        for j in range(L):
            lattice[i,j] = random.uniform(0, 2*np.pi) #Random uniform is a good choice
            # since we're now allowing any value change between 0 and 2pi
            # as opposed to the metropolis where we were restricted to two choices
            
    return lattice
            

def XY_Energy(spins):
    """
    The XY energy of the 2D Lattice
    """
    energy = 0
    
    for i in range(L):
        for j in range(L):
            spin = spins[i,j]
            
            r_neighbour = spins[i, (j+1) % L]
            d_neighbour = spins[(i+1) % L, j]
            
            # -J * cos (spin of angle - neighbour spin angle)
            energy += -J * (np.cos(spin - r_neighbour) + np.cos(spin - d_neighbour))
    return energy

def XY_Metropolis(spins, T):
    """
    One sweep of the XY
    lattice.
    """
    for _ in range(L*L):
        i = random.randint(0, L-1)
        j = random.randint(0, L-1)
    
    #Four Neighbours
    neighbours_sum = (
        down = spins[(i-1) % L, j] + # Down
        up = spins[(i+1) % L, j] + # Up
        right = spins[i, (j+1) % L] + # Right
        left = spins[i, (j-1) % L]   # Left
    )
    # s_i * s_j takes the form cos(theta_i - theta_j) (Just the dot product of s_i and s_j)
    # so XY Energy = J*cos(theta_i - theta_j)
    # I have chosen to subtract the initial energy of the XY lattice
    # from the new energy post angle change
    # Don't use random.choice for delta_angle since it's not a discrete list
    new_angle = spin + random.uniform(-delta_angle, delta_angle)
    initial_energy = -J * (np.cos(spin - down) + np.cos(spin - up) + np.cos(spin - right) +
                           np.cos(spin - left)
                      )
    
    new_energy = -J * (np.cos(new_angle - down) + np.cos(new_angle - up) +
                       np.cos(new_angle - left) + np.cos(new_angle - right)
                 )
    
    delta_E = new_energy - initial_energy
    
    #Acception or Rejection
    if delta_E <= 0:
        spins[i,j] = new_angle      # If energy decreases, the angle change is accepted
        
    else:
        w = np.exp(-delta_E / T)
        if random.random() < w:
            spins[i,j] = new_angle
        
def XY_sim(T):
    """
    Sweeps of the XY model at
    temperature T
    """
    spins = XY_lattice()
    
    energies = []
    magnetisation = []
    
    for _ in range(N_SWEEPS):
        XY_metropolis(spins, T)
        energy = XY_Energy(spins)
        #We are no longer summing 1's and -1's for magnetization.
        #We are summing the cosine of all the angles of spins in the lattice
        M = np.sum(np.cos(spins) + np.sum(np.sin(spins))
        
        energies.append(energy)
        magnetisation.append(M)
        
    return np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2)

if rank < nproc - 1:
    start = rank * N_SPLIT
    end = start + N_SPLIT
else: 
    start = rank * N_SPLIT
    end = N_SWEEPS
    
for T in temperatures:
    localE = 0
    localM = 0
    localE2 = 0
    
    for _ in range(start,end):
        TotalE, TotalM, TotalE2 = XY_sim(T)
        localE += TotalE
        localM += TotalM
        localE2 += TotalE2
        
    globalE = comm.reduce(localE, op=MPI.SUM, root=0)
    globalM = comm.reduce(localM, op=MPI.SUM, root=0)
    globalE2 = comm.reduce(localE2, op=MPI.SUM, root=0)
    
    if rank == 0:
        Cv = 1/1.38E-23 * T**2 * (globalE2 - E**2)
        print(f"T={T:.2f}, E={globalE:.2f}, M={globalM:.2f}, E2={globalE2:.2f}")
    
    # Frction of the lattice from 10% - 100%
    x_fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    average_spin = []
    for x_frac in x_fracs: # For each % in x_fracs
        x = int(r_frac * L) # Added int because of index error, add it for everything now.
        average = spin_correlation(spins, r)
        average_spin.append(average)
    
    plt.plot(r_fracs, all_corr)
    plt.title(f'XY Model: Spin Correlation vs Fractional Separation, T={T:.2f}, L={L}')
    plt.xlabel('r/L')
    plt.ylabel('Spin Correlation')
    plt.savefig(f'XY_correlation_T{T:.2f}_{nproc}.png', dpi=300)
    plt.close()
    
comm.Barrier()
if rank == 0:
    endtime = time.time()
    print(f"Number of processors: {nproc}")
    print(f"Timing: {endtime - starttime:.4f} seconds")
    
    plt.plot(temperatures, E)
    plt.title(f'XY Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'XYEnergy_{nproc}.png', dpi=300)
    plt.close()
    
    plt.plot(temperatures, M)
    plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
    plt.close()
    
    plt.plot(temperatures, Cv)
    plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
    plt.xlabel('Temperature (kT/J)')
    plt.ylabel('Energy')
    plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
    plt.close()
        

#Spin Correlation
# I think this is asking for the average spin as a fraction of the lattice
# i.e, average spin per x/L

def spin_correlation(spins, x):
    """
    Spin correlation across the lattice for fractional
    distance x/L
    """
    correlation = 0      # Total correlation
    for i in range(L):
        for j in range(L):
            # cos(theta_i - theta_j) at separation x, %L sets the boundary
            correlation += np.cos(spins[i,j] - spins[i, (j+r) % L])
    return correlation / (L*L)         #Total correlation over entire grid

for L in [16, 32, 64, 128, 256]
