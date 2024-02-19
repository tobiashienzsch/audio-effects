import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def dynamic_smoothing(x: np.ndarray, fs, base_frequency=2.0, sensitivity=0.5):
    y = x.copy()

    wc = base_frequency / fs
    inz = 0
    _low1 = 0
    _low2 = 0

    x1 = 5.9948827
    x2 = -11.969296
    x3 = 15.959062

    for idx, _ in enumerate(y):
        low1z = _low1
        low2z = _low2
        bandz = low1z - low2z

        wd = wc + sensitivity * abs(bandz)
        g = min(wd * (x1 + wd * (x2 + wd * x3)), 1)

        _low1 = low1z + g * (0.5 * (y[idx] + inz) - low1z)
        _low2 = low2z + g * (0.5 * (_low1 + low1z) - low2z)
        inz = y[idx]

        y[idx] = _low2

    return y


sr = 1000.0
t = np.linspace(0, 2, int(sr*2))

saw_slow = (signal.sawtooth(2 * np.pi * 1 * t)+1.0)/2
saw_fast = (signal.sawtooth(2 * np.pi * 10 * t))*0.1

sig = np.where(t > 1.5, 1.0, np.where(t > 1.05, 0.5, saw_slow))
sig += np.where((t > 0.5) & (t < 1.0), saw_fast, 0.0)
sig += (np.random.random(t.shape[0]) - 0.5)*0.1

plt.plot(t, sig, label="Singal")
plt.plot(t, dynamic_smoothing(sig, sr, 1.0, 0.5), label="Output (0.5)")
plt.plot(t, dynamic_smoothing(sig, sr, 2.0, 0.2), label="Output (0.2)")
plt.legend()
plt.grid(which="both")
plt.minorticks_on()
plt.show()
