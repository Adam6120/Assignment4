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
N_SWEEPS = 1500                   # Number of sweeps of the metropolis
N_SPLIT = N_SWEEPS // nproc # Splitting work among processors
delta_angle = np.pi
ising_temperatures = np.linspace(1.0, 3.0, 30) #Array of Ising temperatures
xy_temperatures = np.linspace(0.5, 1.5, 30) #Array of XY temperatures
N_BURN = 500
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

def ising_sim(T, L):
    """
    Sweeps of the ising model at a temperature T
    """
    spins = lattice(L)
    for _ in range(N_BURN):
        metropolis(spins, T, L)    # Adding burn in to NOT measure the initial sweeps
                                   #Burn in simply lets the randomly oriented lattice succumb to the effect
                                   # of the temperature we are applying to it
 
    energies = []
    magnetisation = []
  
    for _ in range(N_SWEEPS):
        metropolis(spins, T, L)
        energy = ising_energy(spins, L)
        M = abs(np.sum(spins)) #Magnetisation is the sum of all spins in the lattice.
        
        energies.append(energy)      #Storage of the energy
        magnetisation.append(M)      #Storage of the magnetisation
        
        
    return np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2) # E, M and E^2
    # np.mean(energies**2) gives an error for some reason, np.array fixes it.  
    # These gets caught by localE, localM and localE2
        

# Task 4 XY Model (Angle Incorporation)
# The X-Y model incorporates angles into the spins, so instead of
# spins up or down, they can have an infinite number of variations
# in between 0 and 2pi
def XY_lattice(L):
    """
    Function which creates a lattice filled
    with randomly oriented spins which can
    range from 0 - 2pi
    """
    
    grid = np.zeros((L,L), dtype=float)    #Using float since angles are decimals
    
    for i in range(L):
        for j in range(L):
            grid[i,j] = random.uniform(0, 2*np.pi) #Random uniform is a good choice
            # since we're now allowing any value change between 0 and 2pi
            # as opposed to the metropolis where we were restricted to two choices
            
    return grid


def XY_Energy(spins, L):
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

def XY_Metropolis(spins, T, L):
    """
    One sweep of the XY
    lattice.
    """
    for _ in range(L*L):
        i = random.randint(0, L-1)
        j = random.randint(0, L-1)
        
        #Four Neighbours
        down = spins[(i-1) % L, j]   # Down
        up = spins[(i+1) % L, j]     # Up
        right = spins[i, (j+1) % L]  # Right
        left = spins[i, (j-1) % L]   # Left
    
        # s_i * s_j takes the form cos(theta_i - theta_j) (Just the dot product of s_i and s_j)
        # so XY Energy = J*cos(theta_i - theta_j)
        # I have chosen to subtract the initial energy of the XY lattice
        # from the new energy post angle change
        # Don't use random.choice for delta_angle since it's not a discrete list
        new_angle = spins[i,j] + random.uniform(-delta_angle, delta_angle)
        initial_energy = -J * (np.cos(spins[i,j] - down) + np.cos(spins[i,j] - up) +
                           np.cos(spins[i,j] - right) + np.cos(spins[i,j] - left))
    
        new_energy = -J * (np.cos(new_angle - down) + np.cos(new_angle - up) +
                           np.cos(new_angle - left) + np.cos(new_angle - right))
    
        delta_E = new_energy - initial_energy
    
        #Acception or Rejection
        if delta_E <= 0:
            spins[i,j] = new_angle      # If energy decreases, the angle change is accepted
        
        else:
            w = np.exp(-delta_E / T)
            if random.random() < w:
                spins[i,j] = new_angle


def XY_sim(T, L):
    """
    Sweeps of the XY model at
    temperature T
    """

    spins = XY_lattice(L)
    for _ in range(N_BURN):           #Burn in simply lets the randomly oriented lattice succumb to the effect
                                      # of the temperature we are applying to it
        XY_Metropolis(spins, T, L)

    
    
    energies = []
    magnetisation = []
    
    for _ in range(N_SWEEPS):
        XY_Metropolis(spins, T, L)
        energy = XY_Energy(spins, L)
        #We are no longer summing 1's and -1's for magnetization.
        #We are summing the cosine of all the angles of spins in the lattice
        M = np.sum(np.cos(spins)) + np.sum(np.sin(spins))
        
        energies.append(energy)
        magnetisation.append(M)
        
    return np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2), spins
    # Added spins because it was undefined during the spin corr. part.

def spin_correlation(spins, x, L):
    """
    Spin correlation across the lattice for fractional
    distance x/L
    """
    correlation = 0      # Total correlation
    for i in range(L):
        for j in range(L):
            # cos(theta_i - theta_j) at separation x, %L sets the boundary
            correlation += np.cos(spins[i,j] - spins[i, (j+x) % L])
    return correlation / (L*L)         #Total correlation over entire grid


    
# Task 3: Ising Temperature
comm.Barrier()

if rank == 0:
    ising_starttime = time.time()

for L in [16, 32, 64, 128, 256]:
    E_results = []
    M_results = []
    E2_results = []
    Cv_results = []

    for T in ising_temperatures:

    
        
        TotalE, TotalM, TotalE2 = ising_sim(T, L)
        
        globalE = comm.reduce(TotalE, op=MPI.SUM, root=0)
        globalM = comm.reduce(TotalM, op=MPI.SUM, root=0)
        globalE2 = comm.reduce(TotalE2, op=MPI.SUM, root=0)
    
        if rank == 0:
            meanE = globalE / nproc / (L*L)          # Average values and now calcualates per spin
            meanM = abs(globalM / nproc / (L*L))
            meanE2 = globalE2 / nproc / (L*L)**2     # Wasn't squaring E2 properly
            Cv = 1/T**2 * (meanE2 - meanE**2) * (L*L) #Got rid of kB because we want units of kB/J
        
            E_results.append(meanE)
            M_results.append(meanM)
            E2_results.append(meanE2)
            Cv_results.append(Cv)
            print(f"T={T:.2f}, E={meanE:.2f}, M={meanM:.2f}, E2={meanE2:.2f}, Cv={Cv:.2f}")
    
    comm.Barrier()
    if rank == 0:
        ising_endtime = time.time()
        print(f"Ising Model - Lattice Size: {L}, Number of processors: {nproc}---------------------------------------")
        print(f"Ising Model - Lattice Size: {L}, Timing: {ising_endtime - ising_starttime:.2f} seconds. ------------- ")
     
        plt.plot(ising_temperatures, E_results)
        plt.title(f'Ising Model: Energy vs Temperature for Lattice size: {L}, Processors: {nproc}')
        plt.xlabel('Temperature (kT/J)')
        plt.ylabel('Energy')
        plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
        plt.close()
    
        plt.plot(ising_temperatures, M_results)
        plt.title(f'Ising Model: Magnetization vs Temperature for Lattice size: {L}, Processors: {nproc}')
        plt.xlabel('Temperature (kT/J)')
        plt.ylabel('Magnetization')
        plt.savefig(f'IsingMagnetization_{nproc}.png', dpi=300)
        plt.close()
    
        plt.plot(ising_temperatures, Cv_results)
        plt.title(f'Ising Model: Specific Heat Capacity vs Temperature for Lattice size: {L}, Processors: {nproc}')
        plt.xlabel('Temperature (kT/J)')
        plt.ylabel('Specific Heat Capacity Cv')
        plt.savefig(f'IsingHeatCapacity_{nproc}.png', dpi=300)
        plt.close()

#XY Model-----------------------------------------------------------------------------------------


comm.Barrier()
if rank == 0:
    xy_starttime = time.time()
    

for L in [16, 32, 64, 128, 256]:
    E_results = []
    M_results = []
    E2_results = []
    Cv_results = []

    if rank == 0:
        x_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        correlation_results = []
    
    for T in xy_temperatures:

    
        TotalE, TotalM, TotalE2, spins = XY_sim(T, L)

        
        globalE = comm.reduce(TotalE, op=MPI.SUM, root=0)
        globalM = comm.reduce(TotalM, op=MPI.SUM, root=0)
        globalE2 = comm.reduce(TotalE2, op=MPI.SUM, root=0)
    
        if rank == 0:
            meanE = globalE / nproc / (L*L) # /L*L for per spin
            meanM = abs(globalM / nproc / (L*L)) # Adding absolute value because getting negative magnetisation
            meanE2 = globalE2 / nproc / (L*L)**2
            Cv = (1 / T**2) * (meanE2 - meanE**2) *(L*L)  # Forgot to convert Cv to per spin
            # Hopefully this will fix the super low Cv values all across
        
            E_results.append(meanE)
            M_results.append(meanM)
            E2_results.append(meanE2)
            Cv_results.append(Cv)
        
        
            print(f"T={T:.2f}, E={meanE:.2f}, M={meanM:.2f}, E2={meanE2:.2f}, Cv={Cv:.2f}")
        
            correlation_T = []                   # Correlation for each T, empty array


             # Spin Correlations for fractional pieces of the entire lattice for each T

            for x_fraction in x_fractions:
                x = int(x_fraction * L)          # Fraction in terms of lattice size
                average = spin_correlation(spins, x, L)  # Correlation between neighours
                correlation_T.append(average)

            correlation_results.append(correlation_T)


    comm.Barrier()
    if rank == 0:
        xy_endtime = time.time()
        print(f"XY Model - Lattice Size: {L}, Number of processors: {nproc}------------------------------------------")
        print(f"XY Model - Lattice Size: {L}, Timing: {xy_endtime - xy_starttime:.2f} seconds------------------------ ")

            
    
if rank == 0:
    # Specific HEAT plot
    plt.plot(xy_temperatures, Cv_results, label=f'L={L}')
    plt.title(f'XY Model: Specific Heat vs Temperature, Processors: {nproc}')
    plt.xlabel('Temperature (kBT/J)')
    plt.ylabel('Cv')
    plt.legend()
    plt.savefig(f'XY_Cv_{nproc}.png', dpi=300)
    plt.close()
    
    # Selecting lowest, middle and highest temperature values from correlation_results
    # This array will have 30 entries, double checked with len(correlation_results)
    plt.plot(x_fractions, correlation_results[0],  label=f'T={xy_temperatures[0]:.2f}K')   # Low T: 
    plt.plot(x_fractions, correlation_results[14], label=f'T={xy_temperatures[14]:.2f}K')  # Mid T:
    plt.plot(x_fractions, correlation_results[29], label=f'T={xy_temperatures[29]:.2f}K')  # High T:  
    plt.title(f'XY Model: Spin Correlation vs Fractional Separation, L={L}, Processors: {nproc}')
    plt.xlabel('x/L')
    plt.ylabel('Spin Correlation')
    plt.legend()
    plt.savefig(f'XY_correlation_{nproc}.png', dpi=300)
    plt.close()
