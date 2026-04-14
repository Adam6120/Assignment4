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


def lattice(lattice_size):
    """
    Function which creates a lattice filled with
    randomly oriented spins +- 1
    """
    grid = np.zeros((lattice_size, lattice_size), dtype=int)

    for i in range(lattice_size):      # Using two loops since we're in 2D
        for j in range(lattice_size):  # Using two loops since we're in 2D
            grid[i, j] = random.choice([-1, 1]) # Equal chance of spin being -1 or 1

    return grid          # Returns a L x L grid of random spins

def ising_energy(
    spins, lattice_size
): # Ising Energy which is the energy of a ferromagnetic system
   # based on the spins of neighbouring particles is equal to the
   # negative of the interaction strength multiplied by the sum of
   # the product of spins of the neighburing atom
    """
    The ising energy of the 2D Lattice
    """
    energy = 0             # Energy initially zero, gets updated throughout the loop.

    for i in range(lattice_size):
        for j in range(lattice_size):
            # Extracts each spin from the lattice
            # The <i,j> in the formula means nearest neighours
            # (i,j) is the spin at location (i,j) which is
            # multiplied by a neighbouring spin such that
            r_neighbour = spins[i, (j+1) % lattice_size]
            d_neighbour = spins[(i+1) % lattice_size, j]

            energy += -spins[i, j] * (r_neighbour + d_neighbour)
    return energy                            # E = -(spin * neighbour spin) (summed over all (i,j))
                                             # So confusingly, s_i is the spin at (i,j)
                                             # and s_j is the spin at (i,j+1)

# a b c d
# e f g h
# where (i,j) = a  s_i = a, s_j = (b, right neighbour) and (e, down neighbour)
# i j k l            so the ising energy sums those: a*b + a*e for that location (i,j)


#For the metropolis algorithm, we're now interested in four neighbours per point.
def metropolis(spins, temperature, lattice_size):
    """
    One sweep of the lattice
    Using equation 9.15 from notes
    """
    for _ in range(lattice_size*lattice_size):     # One full sweep is L*L
        i = random.randint(0, lattice_size-1)      # Picking random location (i,j) to flip
        j = random.randint(0, lattice_size-1)

#Taking the four neighbours
        neighbours_sum = (
            spins[(i-1) % lattice_size, j] + # Down
            spins[(i+1) % lattice_size, j] + # Up
            spins[i, (j+1) % lattice_size] + # Right
            spins[i, (j-1) % lattice_size]   # Left
        )
        delta_e = 2 * J * spins[i, j] * neighbours_sum  # Equation 9.15 fledged out

        #Acceptance or Rejection
        if delta_e <= 0:
            spins[i, j] = -1 * spins[i, j] # Flip the sign
        else:
            w = np.exp(
                -delta_e / temperature
            )   #If E > 0, this is the probability of acceptance
            if random.random() < w:
                spins[i, j] = -1 * spins[i, j] # Flip the sign

def ising_sim(temperature, lattice_size):
    """
    Sweeps of the ising model at a temperature T
    """
    spins = lattice(lattice_size)
    for _ in range(N_BURN):
        metropolis(spins, temperature, lattice_size)

# Adding burn in to NOT measure the initial sweeps
# Burn in simply lets the randomly oriented lattice succumb to the effect
# of the temperature we are applying to it

    energies = []
    magnetisation = []

    for _ in range(N_SWEEPS):
        metropolis(spins, temperature, lattice_size)
        energy = ising_energy(spins, lattice_size)
        mag = abs(np.sum(spins)) #Magnetisation is the sum of all spins in the lattice.

        energies.append(energy)      #Storage of the energy
        magnetisation.append(mag)      #Storage of the magnetisation


    return (
        np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2)
    ) # E, M and E^2
    # np.mean(energies**2) gives an error for some reason, np.array fixes it.
    # These gets caught by localE, localM and localE2


# Task 4 XY Model (Angle Incorporation)
# The X-Y model incorporates angles into the spins, so instead of
# spins up or down, they can have an infinite number of variations
# in between 0 and 2pi
def xy_lattice(lattice_size):
    """
    Function which creates a lattice filled
    with randomly oriented spins which can
    range from 0 - 2pi
    """

    # Using float since angles are decimals
    grid = np.zeros((lattice_size, lattice_size), dtype=float)

    for i in range(lattice_size):
        for j in range(lattice_size):
            grid[i, j] = random.uniform(0, 2*np.pi) #Random uniform is a good choice
            # since we're now allowing any value change between 0 and 2pi
            # as opposed to the metropolis where we were restricted to two choices

    return grid


def xy_energy(spins, lattice_size):
    """
    The XY energy of the 2D Lattice
    """
    energy = 0

    for i in range(lattice_size):
        for j in range(lattice_size):
            spin = spins[i, j]

            r_neighbour = spins[i, (j+1) % lattice_size]
            d_neighbour = spins[(i+1) % lattice_size, j]

            # -J * cos (spin of angle - neighbour spin angle)
            energy += -J * (
                np.cos(spin - r_neighbour) + np.cos(spin - d_neighbour)
            )
    return energy

def xy_metropolis(spins, temperature, lattice_size):
    """
    One sweep of the XY
    lattice.
    """
    for _ in range(lattice_size*lattice_size):
        i = random.randint(0, lattice_size-1)
        j = random.randint(0, lattice_size-1)

        #Four Neighbours
        down = spins[(i-1) % lattice_size, j]   # Down
        up = spins[(i+1) % lattice_size, j]     # Up
        right = spins[i, (j+1) % lattice_size]  # Right
        left = spins[i, (j-1) % lattice_size]   # Left

        # s_i * s_j takes the form cos(theta_i - theta_j)
        # (Just the dot product of s_i and s_j)
        # so XY Energy = J*cos(theta_i - theta_j)
        # I have chosen to subtract the initial energy of the XY lattice
        # from the new energy post angle change
        # Don't use random.choice for delta_angle since it's not a discrete list
        new_angle = spins[i, j] + random.uniform(-delta_angle, delta_angle)
        initial_energy = -J * (
            np.cos(spins[i, j] - down) + np.cos(spins[i, j] - up) +
            np.cos(spins[i, j] - right) + np.cos(spins[i, j] - left)
        )

        new_energy = -J * (np.cos(new_angle - down) + np.cos(new_angle - up) +
                           np.cos(new_angle - left) + np.cos(new_angle - right))

        delta_e = new_energy - initial_energy

        #Acception or Rejection
        if delta_e <= 0:
            spins[i, j] = new_angle      # If energy decreases, the angle change is accepted

        else:
            w = np.exp(-delta_e / temperature)
            if random.random() < w:
                spins[i, j] = new_angle


def xy_sim(temperature, lattice_size):
    """
    Sweeps of the XY model at
    temperature T
    """

# Burn in simply lets the randomly oriented lattice succumb to the effect
# Of the temperature we are applying to it

    spins = xy_lattice(lattice_size)
    for _ in range(N_BURN):


        xy_metropolis(spins, temperature, lattice_size)



    energies = []
    magnetisation = []

    for _ in range(N_SWEEPS):
        xy_metropolis(spins, temperature, lattice_size)
        energy = xy_energy(spins, lattice_size)
        #We are no longer summing 1's and -1's for magnetization.
        #We are summing the cosine of all the angles of spins in the lattice
        mag = np.sum(np.cos(spins)) + np.sum(np.sin(spins))

        energies.append(energy)
        magnetisation.append(mag)

    return (
        np.mean(energies), np.mean(magnetisation), np.mean(np.array(energies)**2), spins
    )
    # Added spins because it was undefined during the spin corr. part.

def spin_correlation(spins, separation, lattice_size):
    """
    Spin correlation across the lattice for fractional
    distance x/L
    """
    correlation = 0      # Total correlation
    for i in range(lattice_size):
        for j in range(lattice_size):
            # cos(theta_i - theta_j) at separation x, %L sets the boundary
            correlation += np.cos(
                spins[i, j] - spins[i, (j+separation) % lattice_size]
            )
    return correlation / (lattice_size*lattice_size)         #Total correlation over entire grid



def main():
    """
    Wrapping all executable code
    in a main function to prevent
    the creation of global
    variables being flagged
    """
# Used highlight = tab to auto indent eveyrthing

# Task 3: Ising Temperature
    comm.Barrier()

    if rank == 0:
        ising_starttime = time.time()

    for lattice_size in [16]:
        e_results = []
        m_results = []
        e2_results = []
        cv_results = []

        for temperature in ising_temperatures:



            total_e, total_m, total_e2 = ising_sim(temperature, lattice_size)

            global_e = comm.reduce(total_e, op=MPI.SUM, root=0)
            global_m = comm.reduce(total_m, op=MPI.SUM, root=0)
            global_e2 = comm.reduce(total_e2, op=MPI.SUM, root=0)

            # Average values and now calcualates per spin

            if rank == 0:
                mean_e = global_e / nproc / (lattice_size*lattice_size)
                mean_m = abs(global_m / nproc / (lattice_size*lattice_size))
                mean_e2 = global_e2 / nproc / (lattice_size*lattice_size)**2
                cv = 1/temperature**2 * (mean_e2 - mean_e**2) * (lattice_size*lattice_size)

                e_results.append(mean_e)
                m_results.append(mean_m)
                e2_results.append(mean_e2)
                cv_results.append(cv)
                print(
                    f"T={temperature:.2f}, E={mean_e:.2f}, M={mean_m:.2f}, "
                    f"E2={mean_e2:.2f}, Cv={cv:.2f}"
                )

        comm.Barrier()
        if rank == 0:
            ising_endtime = time.time()
            print(
                f"Ising Model - Lattice Size: {lattice_size}, Number of processors: "
                f"{nproc} ---------------------------------------"
            )

            print(
                f"Ising Model - Lattice Size: {lattice_size}, Timing: "
                f"{ising_endtime - ising_starttime:.2f} seconds. -------------"
            )

            plt.plot(ising_temperatures, e_results, color='black')
            plt.title(
                    f"Ising Model: Energy vs Temperature "
                    f"for Lattice size: {lattice_size}, Processors: {nproc}"
            )
            plt.xlabel('Temperature (kT/J)')
            plt.ylabel('Energy')
            plt.grid(True)
            plt.savefig(f'IsingEnergy_{nproc}.png', dpi=300)
            plt.close()

            plt.plot(ising_temperatures, m_results, color='black')
            plt.title(
                    f"Ising Model: Magnetization vs Temperature "
                    f"for Lattice size: {lattice_size}, Processors: {nproc}"
            )
            plt.xlabel('Temperature (kT/J)')
            plt.ylabel('Magnetization')
            plt.grid(True)
            plt.savefig(f'IsingMagnetization_{nproc}.png', dpi=300)
            plt.close()

            plt.plot(ising_temperatures, cv_results, color='black')
            plt.title(
                    f"Ising Model: Specific Heat Capacity vs Temperature "
                    f"for Lattice size: {lattice_size}, Processors: {nproc}"
            )
            plt.xlabel('Temperature (kT/J)')
            plt.ylabel('Specific Heat Capacity Cv')
            plt.grid(True)
            plt.savefig(f'IsingHeatCapacity_{nproc}.png', dpi=300)
            plt.close()

    #XY Model--------------------------------------------------------------------------------------


    comm.Barrier()
    if rank == 0:
        xy_starttime = time.time()


    for lattice_size in [16]:
        e_results = []
        m_results = []
        e2_results = []
        cv_results = []

        if rank == 0:
            x_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            correlation_results = []

        for temperature in xy_temperatures:


            total_e, total_m, total_e2, spins = xy_sim(temperature, lattice_size)


            global_e = comm.reduce(total_e, op=MPI.SUM, root=0)
            global_m = comm.reduce(total_m, op=MPI.SUM, root=0)
            global_e2 = comm.reduce(total_e2, op=MPI.SUM, root=0)

            if rank == 0:
                mean_e = global_e / nproc / (lattice_size*lattice_size) # /L*L for per spin
                mean_m = abs(global_m / nproc / (lattice_size*lattice_size))
                mean_e2 = global_e2 / nproc / (lattice_size*lattice_size)**2
                cv = (1 / temperature**2) * (mean_e2 - mean_e**2) *(lattice_size*lattice_size)
                # Hopefully this will fix the super low Cv values all across
                # Adding absolute value because getting negative magnetisation

                e_results.append(mean_e)
                m_results.append(mean_m)
                e2_results.append(mean_e2)
                cv_results.append(cv)


                print(
                    f"T={temperature:.2f}, E={mean_e:.2f}, M={mean_m:.2f}, "
                    f"E2={mean_e2:.2f}, Cv={cv:.2f}"
                )

                correlation_temperature = []                  # Correlation for each T, empty array


                 # Spin Correlations for fractional pieces of the entire lattice for each T

                # Fraction in terms of lattice size
                # Correlation between neighours
                for x_fraction in x_fractions:
                    separation = int(x_fraction * lattice_size)
                    average = spin_correlation(spins, separation, lattice_size)
                    correlation_temperature.append(average)

                correlation_results.append(correlation_temperature)


        comm.Barrier()
        if rank == 0:
            xy_endtime = time.time()
            print(
                f"XY Model - Lattice Size: {lattice_size}, Number of processors: {nproc}"
                f"------------------------------------------"
            )

            print(
                f"XY Model - Lattice Size: {lattice_size}, Timing: "
                f"{xy_endtime - xy_starttime:.2f} seconds ------------------------"
            )

    if rank == 0:
        # Specific HEAT plot
        plt.plot(
            xy_temperatures, cv_results, label=f'L={lattice_size}', color='black'
        )
        plt.title(f'XY Model: Specific Heat vs Temperature, Processors: {nproc}')
        plt.xlabel('Temperature (kBT/J)')
        plt.ylabel('Cv')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'XY_Cv_{nproc}.png', dpi=300)
        plt.close()

        # Selecting lowest, middle and highest temperature values from correlation_results
        # This array will have 30 entries, double checked with len(correlation_results)
        plt.plot(
            x_fractions, correlation_results[0], label=f'T={xy_temperatures[0]:.2f}K',
            color='black'
        )   # Low T:
        plt.plot(
            x_fractions, correlation_results[14], label=f'T={xy_temperatures[14]:.2f}K',
            color='red'
        )  # Mid T:
        plt.plot(
            x_fractions, correlation_results[29], label=f'T={xy_temperatures[29]:.2f}K',
            color='lightblue'
        )  # High T:
        plt.title(
            f'XY Model: Spin Correlation vs Fractional Separation, '
            f'L={lattice_size}, Processors: {nproc}'
        )
        plt.xlabel('x/L')
        plt.ylabel('Spin Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'XY_correlation_{nproc}.png', dpi=300)
        plt.close()


if __name__ == "__main__":
    main()

# Line to fix code not running, placed at the bottom to let all variables be defined
# Before running the code.
