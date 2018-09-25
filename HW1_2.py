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

    def __init__(self, a, b, c, d, V):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.resetV = V
        self.V = self.resetV
        self.u = self.b * self.V
        print(self.a,self.b,self.c,self.d,self.V,self.u)

    def step(self, I, tau):
        self.V = self.V + tau * (0.04 * self.V**2 + 5 * self.V + 140 - self.u + I)
        self.u = self.u + tau * self.a * (self.b* self.V - self.u)

        if self.V > 30:
            self.V = self.c
            self.u = self.u + self.d
            # Return if a spike occured and VV
            return (1, 30)
        else:
            return (0, self.V)

    #Helper functions for testing
    def getV(self):
        return self.V

    def getU(self):
        return self.u

    # Reset neuron for each trial
    def reset(self):
        self.V = self.resetV


def plotFigs(tspan, v_plot, I, R):

    # Plot VV figure
    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot[n])
    plt.show()

    # Plot R figure
    plt.plot(I,R)
    plt.show()

def main():

    # Simulation runs for 1000 steps
    steps = 1000

    N1 = Neuron(0.02, 0.25, -65, 6, -64)
    N2 = Neuron(0.02, 0.25, -65, 6, -64)

    tau = 0.25
    tspan = np.arange(0,steps+tau,tau) # Arange understeps by tau

    T1 = 50  # Time at which the step input rises.

    w = 80  # Weight between neurons

    # All of the 81 cases of I to be tested.
    Input = np.linspace(0,20,81)
    Inplot = [1, 5, 10, 15, 20]
    R1 = np.zeros(Input.shape) # preallocate R with the same length as I.
    R2 = np.zeros(Input.shape) # preallocate R with the same length as I.

    v_plot1 = []
    v_plot2 = []
    uplot1 = []
    uplot2 = []
    for istep, i in enumerate(Input):
        print(istep)
        #input("pause")

        spike_counter1 = 0
        spike_counter2 = 0

        VV1 = np.zeros(tspan.shape) # preallocate VV with the same length as tspan.
        N1.reset()  # Reset neuron to inital V

        VV2 = VV1.copy() # Deep copy of VV1
        N2.reset()
        uu1=[]
        uu2=[]
        for tstep, t in enumerate(tspan):

            if t > T1:
                Ia = i  # This is the input changed in part 1.
            else:
                Ia = 0

            (Spike1, VV1[int(tstep)]) = N1.step(Ia, tau)
            (Spike2, VV2[int(tstep)]) = N2.step(Spike1*w, tau)

            uu1.append(N1.getU())
            uu2.append(N2.getU())


            if Spike1 and tstep >= 800.0:
                spike_counter1 += 1
            if Spike2 and tstep >= 800.0:
                spike_counter2 += 1

        if i in Inplot:
            # Store VV to be plotted if in the requested list of I.
            v_plot1.append(VV1)
            v_plot2.append(VV2)
            uplot1.append(uu1)
            uplot2.append(uu2)

        # Calculate spike rate, R
        R1[int(istep)]=spike_counter1/800.0
        R2[int(istep)]=spike_counter2/800.0

    # Plot VV figure 1
    f1 = 10
    f2 = f1-2

    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot1[n])
        plt.xlabel('Time Step', fontsize=f1)
        plt.ylim((-90, 40))
        plt.xlim((0, tspan[-1]))
        plt.ylabel('$V_{mem}$' , fontsize=f1)
        plt.tick_params(labelsize=f2)
        plt.title('Regular Spiking: $I_A$={}'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.925, wspace=5, hspace=0.75)
    plt.suptitle('HW1 pt. 2.1: Neuron A\'s Response Over Time', fontsize=f1+2)
    plt.show()

    # Plot VV figure 2
    for n in range(5):
        plt.subplot(5,1,n+1)
        plt.plot(tspan, v_plot2[n])
        plt.xlabel('Time Step', fontsize=f1)
        plt.ylim((-90, 40))
        plt.xlim((0, tspan[-1]))
        plt.ylabel('$V_{mem}$' , fontsize=f1)
        plt.tick_params(labelsize=f2)
        plt.title('Regular Spiking: $I_A$={}'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.95, top=0.925, wspace=5, hspace=0.75)
    plt.suptitle('HW1 pt. 2.2: Neuron A\'s Response Over Time', fontsize=f1+2)
    plt.show()


    # Plot R figure
    plt.scatter(R1, R2)
    #plt.plot(R1, R2)
    plt.grid(True)
    plt.xlabel('R $I_A$', fontsize=f1)
    plt.ylim((-0.001, max(R2)+0.001))
    plt.xlim((-0.001, max(R1)+0.001))
    plt.ylabel('R $I_B$' , fontsize=f1)
    plt.tick_params(labelsize=f2)
    plt.title('Mean Spike Rate of Neuron B vs. Mean Spike Rate of Neuron A'.format(Inplot[n]), fontsize=f1, fontweight='bold')

    plt.show()

if __name__ == '__main__':
    main()
