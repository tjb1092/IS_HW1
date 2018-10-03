import numpy as np
import matplotlib.pyplot as plt
"""
    This python file replicates the given MATLAB file paper by
                Izhikevich E.M. (2004)
    Which Model to Use For Cortical Spiking Neurons?
    use MATLAB R13 or later. November 2003. San Diego, CA

    Modified by Ali Minai
    Modified again by Tony Bailey for HW 1
"""

class Neuron:
    """
    Izhikevich Neuron Model:
    I defined the neuron as a class so that I could easily replicated its
    behavior in the two neuron system.
    """

    def __init__(self, a, b, c, d, V):
        # Assign neuron parameters to its own internal variables
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.resetV = V
        self.V = self.resetV
        self.u = self.b * self.V
        print(self.a,self.b,self.c,self.d,self.V,self.u)

    def step(self, I, tau):
        # Perform the differential equation calculation via Euler's Method
        self.V = self.V + tau * (0.04 * self.V**2 + 5.0 * self.V + 140.0 - self.u + I)
        self.u = self.u + tau * self.a * (self.b* self.V - self.u)

        # If the membrane voltage is greather than 30, generate a spike
        if self.V > 30:
            self.V = self.c
            self.u = self.u + self.d
            # Return if a spike occured and V
            return (1, 30)
        else:
            return (0, self.V)

    # Reset neuron for each trial
    def reset(self):
        self.V = self.resetV


def main():
    # Simulation runs for 1000 steps
    steps = 1000

    # Instantiate Regular Spiking Neuron
    N1 = Neuron(0.02, 0.25, -65.0, 6.0, -64.0)

    tau = 0.25  # time step length
    tspan = np.arange(0,steps+tau,tau) # Arange understeps by tau

    T1 = 50  # Time at which the step input rises.

    # All of the 81 cases of I to be tested.
    Input = np.linspace(0,20,81)

    Inplot = [1, 5, 10, 15, 20]  # list of values of I that will be plotted.
    R = np.zeros(Input.shape) # preallocate R with the same length as I.

    v_plot = []  # Initialize a list for the series to be plotted.

    for istep, i in enumerate(Input):

        print(istep)
        spike_counter = 0  # Restart spike counter
        VV = np.zeros(tspan.shape) # preallocate VV with the same length as tspan.

        N1.reset()  # Reset neuron to initial -64mV for the next trial.

        for tstep, t in enumerate(tspan):
            if t > T1:
                I = i  # This is the input changed in part 1.
            else:
                I = 0.0

            # Step the neuron forward a time-step.
            (Spike, VV[int(tstep)]) = N1.step(I, tau)

            if Spike and t >= 200.0:
                # If there was a spike and it is the last 800 steps,
                # Increment the spike counter.
                spike_counter+=1


        if i in Inplot:
            # Store VV to be plotted if in the requested list of I.
            v_plot.append(VV)

        # Calculate spike rate, R
        R[int(istep)]=spike_counter/800.0


    # Plot VV figure
    f1 = 10
    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot[n])
        plt.xlabel('Time Step', fontsize=f1)
        plt.ylim((-90, 40))
        plt.xlim((0, tspan[-1]))
        plt.ylabel('$V$' , fontsize=f1)
        plt.tick_params(labelsize=f1-2)
        plt.title('Regular Spiking: I={}'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.925, wspace=5, hspace=0.75)
    plt.suptitle('HW1 pt. 1.1: Neuron\'s Response Over Time', fontsize=f1+2)
    plt.show()

    # Plot R figure
    plt.plot(Input,R)
    plt.xlabel('External Input: I')
    plt.xlim((0, 20))
    plt.ylim((0, max(R)))
    plt.ylabel('Mean Spike-rate: R')
    plt.title('HW1 pt. 1.2: Mean Spike-rate R vs. External Input I'.format(Inplot[n]))
    plt.show()

if __name__ == '__main__':
    # Run the main function.
    main()
