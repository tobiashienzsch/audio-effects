import numpy as np

# Circuit parameters
R = 100.0   # Resistance of resistor (Ohms)
C = 0.01   # Capacitance of capacitor (Farads)
V_T = 0.0257  # Thermal voltage (Volts)

# Initial conditions
v_in = 5.2   # Input voltage (Volts)
v_out = 5.0  # Output voltage (Volts)
i_c = 0.0    # Capacitor current (Amps)

# Time step
dt = 0.001   # Time step (Seconds)

# Number of time steps
num_steps = 10

# Compute the state of the circuit at each time step
for t in range(num_steps):
    # Compute the capacitor current
    i_c += dt * (v_in - v_out) / (R * C)
    # Compute the output voltage
    v_out = v_in + i_c * R
    # Compute the diode current
    i_d = np.clip(i_c, 0.0, np.inf)
    # Compute the voltage drop across the diode
    v_d = V_T * np.log(1.0 + i_d / V_T)
    # Compute the final output voltage
    v_out += v_d
    # Print the output voltage
    print(v_out)
