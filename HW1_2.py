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
a,b,c,d = 0.02, 0.25, -65, 6

V = -64
u = b*V

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
    VV = np.zeros(tspan.shape) # preallocate VV with the same length as tspan.
    uu = np.zeros(tspan.shape) # preallocate uu with the same length as tspan.
    V = -64
    for tstep, t in enumerate(tspan):

        if t > T1:
            I = i  # This is the input changed in part 1.
        else:
            I = 0
        #input("pause")
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

    if i in [1, 5, 10, 15, 20]:
        # Store VV to be plotted if in the requested list of I.
        v_plot.append(VV)

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
