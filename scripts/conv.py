import numpy as np

# Generate a synthetic audio signal with noise
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
signal = np.sin(2 * np.pi * 440 * t) + np.random.normal(scale=0.1, size=len(t))

# Design a lowpass FIR filter
cutoff_frequency = 1000  # Hz
filter_length = 31
window = 'hamming'
fir_filter = signal.firwin(filter_length, cutoff_frequency, nyq=sample_rate / 2, window=window)

# Set the frame size and overlap
frame_size = 1024
overlap = 512

# Compute the number of frames
num_frames = len(signal) // overlap - 1

# Initialize the output signal
output = np.zeros_like(signal)

# Filter the signal using overlapping frames
for i in range(num_frames):
    # Get the current frame
    start = i * overlap
    stop = start + frame_size
    frame = signal[start:stop]

    # Apply the FIR filter to the frame using convolution
    filtered_frame = np.convolve(frame, fir_filter, mode='valid')

    # Overlap and add the filtered frame to the output signal
    output[start:start+len(filtered_frame)] += filtered_frame

# Normalize the output signal
output /= np.max(np.abs(output))
