import numpy as np  # Python analog to MATLAB's arrays
import matplotlib.pyplot as plt  # Python analog to MATLAB's plotting functions
from HW1_1 import Neuron  # Import neuron class for this problem

"""
    This python file replicates the given MATLAB file influenced by
                Izhikevich E.M. (2004)
    Which Model to Use For Cortical Spiking Neurons?
    use MATLAB R13 or later. November 2003. San Diego, CA

    Modified by Ali Minai
    Modified again by Tony Bailey for HW 1 pt 2.

    This program connects two RS neurons together and Neuron A
    is fed a step response I_A. Spike rates are monitored and plotted after
    simulating I_A ranging from 0 to 20.
"""

def main():

    # Simulation runs for 1000 steps
    steps = 1000

    N1 = Neuron(0.02, 0.25, -65.0, 6.0, -64.0)
    N2 = Neuron(0.02, 0.25, -65.0, 6.0, -64.0)

    tau = 0.25  # time step length
    tspan = np.arange(0,steps+tau,tau)  # Note, arange understeps by tau

    T1 = 50  # Time at which the step input rises.

    w = 80.0  # Weight, w, between neurons

    # All of the 81 cases of I to be tested.
    Input = np.linspace(0,20,81)

    R1 = np.zeros(Input.shape)  # preallocate R with the same length as I.
    R2 = np.zeros(Input.shape)  # preallocate R with the same length as I.

    for istep, i in enumerate(Input):

        print(istep)  # Print current iteration.
        # Restart spike counters
        spike_counter1 = 0
        spike_counter2 = 0

        VV1 = np.zeros(tspan.shape)  # preallocate VV with the same length as tspan.
        VV2 = VV1.copy()  # Deep copy of VV1

        N1.reset()  # Reset neuron to inital V
        N2.reset()  # Reset neuron to inital V

        for tstep, t in enumerate(tspan):

            if t > T1:
                Ia = i  # This is the input changed in part 1.
            else:
                Ia = 0.0

            # Step Neuron A.
            (Spike1, VV1[int(tstep)]) = N1.step(Ia, tau)
            # Take the spike output (1 or 0) and multiply it by w.
            (Spike2, VV2[int(tstep)]) = N2.step(Spike1*w, tau)

            # If either neuron spiked and it is the last 800 time steps,
            # Increment the correct spike counter.
            if Spike1 and t >= 200.0:
                spike_counter1 += 1
            if Spike2 and t >= 200.0:
                spike_counter2 += 1

        # Calculate spike rate, R
        R1[int(istep)]=spike_counter1/800.0
        R2[int(istep)]=spike_counter2/800.0

    # Plot N_A-R vs. I_A figure
    fs = 12
    plt.plot(Input,R1)
    plt.xlabel('External Input: $I_A$', fontsize=fs)
    plt.xlim((0, 20))
    plt.ylim((0, max(R1)))
    plt.ylabel('Mean Spike-rate: R', fontsize=fs)
    plt.title('$N_A$\'s Mean Spike-rate R vs. External Input $I_A$', fontsize=fs)
    plt.show()

    # Plot N_B-R vs. I_A figure
    plt.plot(Input,R2)
    plt.xlabel('External Input: $I_A$', fontsize=fs)
    plt.xlim((0, 20))
    plt.ylim((0, max(R2)))
    plt.ylabel('Mean Spike-rate: R', fontsize=fs)
    plt.title('$N_B$\'s Mean Spike-rate R vs. External Input $I_A$', fontsize=fs)
    plt.show()

    # Plot R_N_B vs R_N_A figure
    plt.scatter(R1, R2)
    plt.xlabel('$N_A$\'s Mean Spike Rate: R-$N_A$', fontsize=fs)
    plt.ylim((-0.001, max(R2)+0.001))
    plt.xlim((-0.001, max(R1)+0.001))
    plt.ylabel('$N_B$\'s Mean Spike Rate: R-$N_B$', fontsize=fs)
    plt.title('Mean Spike Rate of $N_B$ vs. Mean Spike Rate of $N_A$', fontsize=fs)

    plt.show()

if __name__ == '__main__':
    main()
