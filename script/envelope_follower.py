import matplotlib.pyplot as plt
import numpy as np

fs = 44100.0
expFactor = -2.0 * np.pi * 1000.0 / fs
t = np.linspace(0.001, 1.0, 1024)
cte = np.exp(expFactor / t)

target = np.log(0.001)
cte2 = np.exp(target / (t * fs * 0.001))


# plt.plot(t, cte, "k-")
# plt.plot(t, cte2, "k:")
# plt.plot(t, t * fs * 0.001)
# plt.grid()
# plt.show()

print(np.exp(np.log(0.01) / (100.0 * fs * 0.001)))
print(10.0**(np.log10(0.01) / (100.0 * fs * 0.001)))


class PeakEnvelopeFollower:
    def __init__(self, attack, release, target, fs):
        self.attack = attack
        self.release = release
        self.target = target
        self.fs = fs

    def process(self, x):
        target = np.log(self.target)
        attack = np.exp(target / (self.attack * self.fs * 0.001))
        release = np.exp(target / (self.release * self.fs * 0.001))

        y = np.copy(x)
        ym1 = 0.0
        for n, _ in enumerate(x):
            env = np.abs(y[n])
            coef = attack if env > ym1 else release
            ym1 = coef * (ym1 - env) + env
            y[n] = ym1
        return y


class TransientShaper:
    def __init__(self, attack, sustain, fs) -> None:
        self.attack = attack
        self.sustain = sustain

        self.attack1 = PeakEnvelopeFollower(1.0, 1000.0, 0.01, fs)
        self.attack2 = PeakEnvelopeFollower(50.0, 1000.0, 0.01, fs)

        max_sustain = 1000.0 * sustain
        self.sustain1 = PeakEnvelopeFollower(1.0, max_sustain, 0.01, fs)
        self.sustain2 = PeakEnvelopeFollower(1.0, max_sustain/20.0, 0.01, fs)

    def process(self, x):
        x_abs = np.abs(x)

        aenv1 = 20*np.log10(self.attack1.process(x_abs)+1e-9)
        aenv2 = 20*np.log10(self.attack2.process(x_abs)+1e-9)
        adiff = np.clip((aenv1-aenv2)*self.attack, -60.0, +60.0)
        again = 10.0**(adiff/20.0)

        senv1 = 20*np.log10(self.sustain1.process(x_abs)+1e-9)
        senv2 = 20*np.log10(self.sustain2.process(x_abs)+1e-9)
        sdiff = (senv1-senv2)*self.sustain
        sgain = 10.0**(sdiff/20.0)

        return np.copy(x)*(again*sgain), again, sgain


fs = 512
t = np.linspace(0.0, 1.0, fs)
pulse = np.zeros(fs)
pulse[fs//100:fs//100*4] = 1.0
pulse = np.sin(2 * np.pi * 30.0 * t)*0.25
pulse[0:fs//8] *= np.linspace(0.0, 1.0, fs//8)
pulse[fs//8*2:fs//8*4] *= 4.0

out, again, sgain = TransientShaper(0.15, 0.75, fs).process(pulse)

fif, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
ax1.plot(t, pulse, label="In")
ax1.plot(t, out, label="Out")
ax1.grid()
ax1.legend()

ax2.plot(t, again, label="Attack")
ax2.plot(t, sgain, label="Sustain")
ax2.grid()
ax2.legend()

plt.show()
