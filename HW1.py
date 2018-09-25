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

# Regular Spiking

# Simulation runs for 1000 steps
steps = 1000


class Neuron:

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = -64
        self.u = self.b * self.V
        print(self.a,self.b,self.c,self.d,self.V,self.u)

    def step(self, I, tau):
        #print(self.V)
        self.V = self.V + tau * (0.04 * self.V**2 + 5 * self.V + 140 - self.u + I)
        self.u = self.u + tau * self.a * (self.b* self.V - self.u)

        if self.V > 30:
            print("Here!")
            return 30 # Return as VV
            self.V = self.c
            self.u = self.u + self.d
        else:
            return self.V

    def reset(self, V):
        self.V = V

        
N1 = Neuron(0.02, 0.25, -65, 6)
#N2 = Neuron(0.02, 0.25, -65, 6)

tau = 0.25
tspan = np.arange(0,steps+tau,tau) # Arange understeps by tau

T1 = 50  # Time at which the step input rises.

# All of the 81 cases of I to be tested.
Input = np.linspace(0,20,81)
R = np.zeros(Input.shape) # preallocate R with the same length as I.

v_plot = []

for istep, i in enumerate(Input):
    print(istep)

    spike_counter=0
    VV1 = np.zeros(tspan.shape) # preallocate VV with the same length as tspan.
    VV2 = VV1.copy() # Deep copy of VV1

    for tstep, t in enumerate(tspan):

        if t > T1:
            I = i  # This is the input changed in part 1.
        else:
            I = 0

        VV1[int(tstep)] = N1.step(I,tau)

        """
        V = V + tau * (0.04 * V**2 + 5 * V + 140 - u + I)
        u = u + tau * a * (b* V - u)

        if V > 30:
            VV[int(tstep)] = 30
            V = c
            u = u + d
            if tstep >= 800.0:
                spike_counter+=1
        else:
            VV[int(tstep)] = V

        uu[int(tstep)] = u
        """

    if i in [1, 5, 10, 15, 20]:
        # Store VV to be plotted if in the requested list of I.
        v_plot.append(VV1)

    # Calculate spike rate, R
    R[int(istep)]=spike_counter/800.0


# Plot VV figure
for n in range(5):
    plt.subplot(5,1,n+1)
    plt.plot(tspan, v_plot[n])
plt.show()

# Plot R figure
plt.plot(Input,R)
plt.show()
