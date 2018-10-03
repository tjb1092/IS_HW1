import numpy as np  # Python analog to MATLAB's arrays
import matplotlib.pyplot as plt  # Python analog to MATLAB's plotting functions
from HW1_1 import Neuron  # Import neuron class for this problem

"""
    This python file replicates the given MATLAB file paper by
                Izhikevich E.M. (2004)
    Which Model to Use For Cortical Spiking Neurons?
    use MATLAB R13 or later. November 2003. San Diego, CA

    Modified by Ali Minai
    Modified again by Tony Bailey for HW 1
"""

def main():

    # Simulation runs for 1000 steps
    steps = 1000

    N1 = Neuron(0.02, 0.25, -65.0, 6.0, -64.0)
    N2 = Neuron(0.02, 0.25, -65.0, 6.0, -64.0)

    tau = 0.25 # time step length
    tspan = np.arange(0,steps+tau,tau) # Note, arange understeps by tau

    T1 = 50  # Time at which the step input rises.

    w = 80.0  # Weight, w, between neurons

    # All of the 81 cases of I to be tested.
    Input = np.linspace(0,20,81)

    Inplot = [1, 5, 10, 15, 20]  # list of values of I that will be plotted.
    R1 = np.zeros(Input.shape) # preallocate R with the same length as I.
    R2 = np.zeros(Input.shape) # preallocate R with the same length as I.

    # Initialize lists for the series to be plotted.
    v_plot1 = []
    v_plot2 = []

    for istep, i in enumerate(Input):

        print(istep)
        # Restart spike counters
        spike_counter1 = 0
        spike_counter2 = 0

        VV1 = np.zeros(tspan.shape) # preallocate VV with the same length as tspan.
        VV2 = VV1.copy() # Deep copy of VV1

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

        if i in Inplot:
            # Store VV to be plotted if in the requested list of I.
            v_plot1.append(VV1)
            v_plot2.append(VV2)

        # Calculate spike rate, R
        R1[int(istep)]=spike_counter1/800.0
        R2[int(istep)]=spike_counter2/800.0

    # Plot VV figure for N_A.
    f1 = 10
    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot1[n])
        plt.xlabel('Time Step', fontsize=f1)
        plt.ylim((-90, 40))
        plt.xlim((0, tspan[-1]))
        plt.ylabel('$V_{mem}$' , fontsize=f1)
        plt.tick_params(labelsize=f1-2)
        plt.title('Regular Spiking: $I_A$={}'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.925, wspace=5, hspace=0.75)
    plt.suptitle('HW1 pt. 2.1: $N_A$\'s Response Over Time', fontsize=f1+2)
    plt.show()

    # Plot VV figure for N_B
    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot2[n])
        plt.xlabel('Time Step', fontsize=f1)
        plt.ylim((-90, 40))
        plt.xlim((0, tspan[-1]))
        plt.ylabel('$V$' , fontsize=f1)
        plt.tick_params(labelsize=f1-2)
        plt.title('Regular Spiking: $I_A$={}'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.925, wspace=5, hspace=0.75)
    plt.suptitle('HW1 pt. 2.2: $N_B$\'s Response Over Time', fontsize=f1+2)
    plt.show()


    # Plot R_N_B vs R_N_A figure
    plt.scatter(R1, R2)
    plt.xlabel('$N_A$\'s Mean Spike Rate: R-$N_A$', fontsize=f1)
    plt.ylim((-0.001, max(R2)+0.001))
    plt.xlim((-0.001, max(R1)+0.001))
    plt.ylabel('$N_B$\'s Mean Spike Rate: R-$N_B$' , fontsize=f1)
    plt.tick_params(labelsize=f1)
    plt.title('HW1 pt. 2.3: Mean Spike Rate of $N_B$ vs. Mean Spike Rate of $N_A$'.format(Inplot[n]),
                fontsize=f1+2)

    plt.show()

if __name__ == '__main__':
    main()
