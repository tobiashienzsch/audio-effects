import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


class ADAA1:
    def __init__(self, f, AD1, TOL=1.0e-5):
        self.TOL = TOL
        self.f = f
        self.AD1 = AD1

    def process(self, x):
        y = np.copy(x)
        x1 = 0.0
        for n, _ in enumerate(x):
            if np.abs(x[n] - x1) < self.TOL:  # fallback
                y[n] = self.f((x[n] + x1) / 2)
            else:
                y[n] = (self.AD1(x[n]) - self.AD1(x1)) / (x[n] - x1)
            x1 = x[n]
        return y


class ADAA2:
    def __init__(self, f, AD1, AD2, TOL=1.0e-5):
        self.TOL = TOL
        self.f = f
        self.AD1 = AD1
        self.AD2 = AD2

    def process(self, x):
        y = np.copy(x)

        def calcD(x0, x1):
            if np.abs(x0 - x1) < self.TOL:
                return self.AD1((x0 + x1) / 2.0)
            return (self.AD2(x0) - self.AD2(x1)) / (x0 - x1)

        def fallback(x0, x2):
            x_bar = (x0 + x2) / 2.0
            delta = x_bar - x0

            if delta < self.TOL:
                return self.f((x_bar + x0) / 2.0)
            return (2.0 / delta) * (self.AD1(x_bar) + (self.AD2(x0) - self.AD2(x_bar)) / delta)

        x1 = 0.0
        x2 = 0.0
        for n, _ in enumerate(x):
            if np.abs(x[n] - x1) < self.TOL:  # fallback
                y[n] = fallback(x[n], x2)
            else:
                y[n] = (2.0 / (x[n] - x2)) * (calcD(x[n], x1) - calcD(x1, x2))
            x2 = x1
            x1 = x[n]
        return y


def plot_fft(x, fs, sm=1.0/24.0):
    fft = 20 * np.log10(np.abs(np.fft.rfft(x) + 1.0e-9))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    return freqs, fft


def process_nonlin(fc, FS, nonlin, gain=10):
    N = 200000
    sin = np.sin(2 * np.pi * fc / FS * np.arange(N))
    y = nonlin(gain * sin)
    freqs, fft = plot_fft(y, FS)
    return freqs, fft


def signum(x):
    return int(0 < x) - int(x < 0)


def hardClip(x):
    return np.where(x > 1, 1, np.where(x < -1, -1, x))


def hardClipAD1(x):
    return x * x / 2.0 if np.abs(x) < 1 else x * signum(x) - 0.5


def hardClipAD2(x):
    return x * x * x / 6.0 if np.abs(x) < 1 else ((x * x / 2.0) + (1.0 / 6.0)) * signum(x) - (x/2)


def tanhClip(x):
    return np.tanh(x)


def tanhClipAD1(x):
    return x - np.log(np.tanh(x) + 1)


def softClip(x):
    return -4*x**3/27 + x


def softClipAD1(x):
    return -x**4/27 + x**2/2


FC = 1244.5
FS = 44100
OS = 2

hardClip_ADAA1 = ADAA1(hardClip, hardClipAD1, 1.0e-5)
hardClip_ADAA2 = ADAA2(hardClip, hardClipAD1, hardClipAD2, 1.0e-5)

freqs_analog, fft_analog = process_nonlin(FC, FS*100, hardClip)
freqs_alias, fft_alias = process_nonlin(FC, FS, hardClip)
freqs_os, fft_os = process_nonlin(FC, FS*OS, hardClip)
freqs_ad1, fft_ad1 = process_nonlin(FC, FS*OS, hardClip_ADAA1.process)
freqs_ad2, fft_ad2 = process_nonlin(FC, FS*OS, hardClip_ADAA2.process)
peak_idxs = signal.find_peaks(fft_analog, 65)[0]

# plt.plot(freqs_analog, fft_analog, '--', c='red', label='Analog')
# plt.plot(freqs_alias, fft_alias, '--', c='green', label='No ADAA')
plt.plot(freqs_os, fft_os, '--', c='orange', label=f'OS{OS}')
plt.plot(freqs_ad1, fft_ad1, 'green', label=f'ADAA1 + OS{OS}')
plt.plot(freqs_ad2, fft_ad2, 'blue', label=f'ADAA2 + OS{OS}')
plt.legend()
plt.scatter(freqs_analog[peak_idxs], fft_analog[peak_idxs], c='r', marker='x')
plt.xlim(0, 20000)
plt.ylim(5)
plt.title('Hard Clipping Distortion')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency [Hz]')
plt.grid()

plt.show()
