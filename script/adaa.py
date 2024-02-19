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


def plot_aliasing(ax, title, freqs_ref, fft_ref, freqs_alias, fft_alias, peaks):
    ax.plot(freqs_ref, fft_ref, '--', c='orange', label='Analog')
    ax.plot(freqs_alias, fft_alias, c='blue', label=title)
    ax.scatter(freqs_ref[peaks], fft_analog[peaks], c='r', marker='x')
    ax.set_title(title)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlim(0, 20000)
    ax.set_ylim(5)
    ax.legend()
    ax.grid()


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
    return np.where(x > 0, +1, np.where(x < 0, -1, 0))
    # return int(0 < x) - int(x < 0)


def softClip3(x):
    return -4*x**3/27 + x


def softClip3AD1(x):
    return -x**4/27 + x**2/2


def softClip3AD2(x):
    return -x**5/135 + x**3/6


def softClip5(x):
    return -((256*x**5)/3125)+x


def softClip5AD1(x):
    return -((128*x**6)/9375)+((x**2)/2)


def softClip5AD2(x):
    return -((128*x**7)/65625) + ((x**3)/6)


def softClipExp(x):
    return np.where(x > 0, 1-np.exp(-x), np.exp(x)-1)


def softClipExpAD1(x):
    return np.where(x <= 0, -x + np.exp(x), x + np.exp(-x))


def softClipExpAD2(x):
    return np.where(x <= 0, (x**2) * -0.5 + np.exp(x), (x**2) * 0.5 - np.exp(-x)+2)


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


def sineClip(x):
    return -np.sin(x*np.pi*0.5)


def sineClipAD1(x):
    return 2*np.cos(x*np.pi*0.5)/np.pi


def sineClipAD2(x):
    return 4*np.sin(x*np.pi*0.5)/(np.pi**2)


def halfWave(x):
    return np.where((x > 0), x, 0)


def halfWaveAD1(x):
    return np.where((x <= 0), 0, (x**2)/2)


def halfWaveAD2(x):
    return np.where((x <= 0), 0, (x**3)/6)


def fullWave(x):
    return np.abs(x)


def fullWaveAD1(x):
    return np.where((x <= 0), (x**2)/-2, (x**2)/2)


def fullWaveAD2(x):
    return np.where((x <= 0), (x**3)/-6, (x**3)/6)


def diode(x):
    return 0.2 * (np.exp(1.79*x) - 1.0)


def diodeAD1(x):
    return -0.2*x + 0.111731843575419*np.exp(1.79*x)


def diodeAD2(x):
    return -0.1*x**2 + 0.0624200243438095*np.exp(1.79*x)


def chebyshevPolynomial(x):
    return 16*x**5 - 20*x**3 + 5*x


def chebyshevPolynomialAD1(x):
    return (8*x**6)/3 - 5*x**4 + (5*x**2)*0.5


def chebyshevPolynomialAD2(x):
    return (8*x**7)/21 - x**5 + (5*x**3)/6


FC = 1244.5
FS = 44100
OS = 2

distortion = "Soft-Clip Exponential"
distortion = "Chebyshev Polynomial"
distortion = "Diode"
distortion = "Full-Wave Rectifier"
distortion = "Soft-Clip (5th Degree)"
distortion = "Tanh-Clip"
distortion = "Hard-Clip"

hardClip_ADAA1 = ADAA1(hardClip, hardClipAD1, 1.0e-5)
hardClip_ADAA2 = ADAA2(hardClip, hardClipAD1, hardClipAD1, 1.0e-5)

freqs_analog, fft_analog = process_nonlin(FC, FS*50, hardClip)
peak_idxs = signal.find_peaks(fft_analog, 65)[0]

freqs_alias, fft_alias = process_nonlin(FC, FS, hardClip)
freqs_os, fft_os = process_nonlin(FC, FS*OS, hardClip)

freqs_ad1, fft_ad1 = process_nonlin(FC, FS, hardClip_ADAA1.process)
freqs_ad1os, fft_ad1os = process_nonlin(FC, FS*OS, hardClip_ADAA1.process)

freqs_ad2, fft_ad2 = process_nonlin(FC, FS, hardClip_ADAA2.process)
freqs_ad2os, fft_ad2os = process_nonlin(FC, FS*OS, hardClip_ADAA2.process)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
    3, 2,
    constrained_layout=True
)

fig.suptitle(distortion)

# TOP-LEFT
plot_aliasing(ax1, "OS1", freqs_analog, fft_analog,
              freqs_alias, fft_alias, peak_idxs)

# TOP-RIGHT
plot_aliasing(ax2, f"OS{OS}", freqs_analog,
              fft_analog, freqs_os, fft_os, peak_idxs)

# MID-LEFT
plot_aliasing(ax3, f"ADAA1", freqs_analog,
              fft_analog, freqs_ad1, fft_ad1, peak_idxs)

# MID-RIGHT
plot_aliasing(ax4, f"ADAA1 + OS{OS}", freqs_analog,
              fft_analog, freqs_ad1os, fft_ad1os, peak_idxs)

# BOTTOM-LEFT
plot_aliasing(ax5, f"ADAA2", freqs_analog,
              fft_analog, freqs_ad2, fft_ad2, peak_idxs)

# BOTTOM-RIGHT
plot_aliasing(ax6, f"ADAA2 + OS{OS}", freqs_analog,
              fft_analog, freqs_ad2os, fft_ad2os, peak_idxs)


plt.show()
