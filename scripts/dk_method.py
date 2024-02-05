import numpy as np

# Circuit parameters
R = 10.0   # Resistance (Ohms)
C = 0.01   # Capacitance (Farads)

# Nonlinear component parameters
V_threshold = 1.0   # Threshold voltage (Volts)
I_max = 1.0         # Maximum current (Amps)

# Initial conditions
v_in = 5.0   # Input voltage (Volts)
v_out = 0.0  # Output voltage (Volts)
i_c = 0.0    # Capacitor current (Amps)

# Time step
dt = 0.001   # Time step (Seconds)

# Number of time steps
num_steps = 100

# Compute the state of the circuit at each time step
for t in range(num_steps):
    # Compute the capacitor current
    i_c += dt * (v_in - v_out) / (R * C)
    # Compute the output voltage
    v_out = v_in + i_c * R
    # Compute the current through the nonlinear component
    i_nl = np.clip(i_c, 0.0, I_max)
    # Compute the voltage drop across the nonlinear component
    v_nl = V_threshold * (1.0 - np.exp(-i_nl))
    # Compute the final output voltage
    v_out += v_nl
    # Print the output voltage
    print(v_out)