import numpy as np


def phase_vocoder(audio, rate, time_stretch_ratio, block_size=1024, window_type='hann'):
    # Create the window function
    window = create_window(block_size, window_type)

    # Calculate the time-stretch factor
    stretch_factor = rate / (rate * time_stretch_ratio)

    # Initialize the output audio
    stretched_audio = np.zeros(len(audio))

    # Process the audio in blocks
    for i in range(0, len(audio), block_size):
        # Get the current block of audio
        block = audio[i:i+block_size]

        # Apply the window function to the block
        windowed_block = block * window

        # Convert the block to the frequency domain
        spectrum = np.fft.rfft(windowed_block)

        # Get the length of the spectrum
        M = len(spectrum)

        # Create an empty array to store the stretched spectrum
        stretched_spectrum = np.zeros(M, dtype=complex)

        # Stretch the spectrum by interpolating the magnitude and phase
        for j in range(M):
            # Calculate the stretched frequency index
            k = int(j * stretch_factor)

            # Interpolate the magnitude and phase
            if k < M:
                stretched_spectrum[j] = spectrum[k]
            else:
                stretched_spectrum[j] = 0

        # Convert the stretched spectrum back to the time domain
        stretched_block = np.fft.irfft(stretched_spectrum)

        # Overlap-add the stretched block to the output audio
        stretched_audio[i:i+block_size] += stretched_block

    # Normalize the audio
    stretched_audio /= np.max(np.abs(stretched_audio))

    return stretched_audio


def create_window(N, window_type='hann'):
    # Create a Hann window
    if window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    # Create a Hamming window
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    # Create a Blackman window
    elif window_type == 'blackman':
        window = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + \
            0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1))
    else:
        raise ValueError('Invalid window type')

    return window
