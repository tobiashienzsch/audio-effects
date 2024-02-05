import numpy as np
import matplotlib.pyplot as plt


def sawtooth(sample_rate, base_frequency):
    # Calculate the period of the sawtooth wave
    period = 1.0 / base_frequency

    # Calculate the length of the wavetable in samples
    wavetable_length = 2048

    # Calculate the actual base frequency of the wavetable
    wavetable_frequency = sample_rate / wavetable_length

    # Generate a sawtooth wavetable with the desired length
    t = np.linspace(0, period, wavetable_length, endpoint=False)
    return t * wavetable_frequency - np.mod(t, 1.0 / wavetable_frequency)


def sin_with_noise(sample_rate, frequency, noise_amplitude=0.1):
    signal = np.sin(2 * np.pi * frequency * np.linspace(0, 1, sample_rate))
    noise = noise_amplitude * np.random.randn(len(signal))
    signal += noise
    return signal


def bandlimit(sample_rate, signal, low, high):
    # Compute the discrete Fourier transform of the wavetable
    dft = np.fft.rfft(signal)

    # Normalize the frequencies
    frequencies = np.fft.rfftfreq(len(wavetable), 1.0 / sample_rate)
    normalized_frequencies = frequencies / (sample_rate / 2)

    # Set the coefficients corresponding to frequencies outside the desired range to zero
    low_frequency = low
    high_frequency = high
    dft[normalized_frequencies < low_frequency] = 0
    dft[normalized_frequencies > high_frequency] = 0

    # Compute the inverse discrete Fourier transform of the modified wavetable
    filtered_wavetable = np.fft.irfft(dft)
    return filtered_wavetable


# Define the wavetable and sample rate
sample_rate = 44100
# wavetable = sawtooth(sample_rate, 440.0)
wavetable = sin_with_noise(sample_rate, 440.0, 0.15)
filtered_wavetable = bandlimit(sample_rate, wavetable, 0, 1.0)


# Print the original and filtered wavetables
print("Original wavetable:", wavetable)
print("Filtered wavetable:", filtered_wavetable)

# Plot the original and filtered wavetables
plt.plot(wavetable, label="Original")
plt.plot(filtered_wavetable, label="Filtered")

plt.legend()
plt.show()


# # Normalize the frequencies
frequencies = np.fft.rfftfreq(len(wavetable), 1.0 / sample_rate)
# normalized_frequencies = frequencies / (sample_rate / 2)

# Plot the spectrum of the wavetable
plt.semilogx(frequencies[4:], np.abs(np.fft.rfft(wavetable))[4:]/sample_rate)
plt.semilogx(frequencies[4:], np.abs(
    np.fft.rfft(filtered_wavetable))[4:]/sample_rate)
plt.xlabel("Normalized frequency")
plt.ylabel("Magnitude")
# Set the tick locations and labels on the x-axis
ticks = [20, 55, 110, 220, 440, 880, 1760, 3520, 7040, 10000]
plt.xticks(ticks, ticks)
plt.show()
