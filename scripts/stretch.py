import numpy as np


def stretch(audio, stretch_factor, block_size, window):
    # Initialize the output array
    stretched_audio = np.zeros_like(audio)

    # Iterate over blocks of the audio signal
    for i in range(0, len(audio), block_size):
        # Window the current block
        block = audio[i:i+block_size] * window

        # Compute the STFT of the current block
        stft = np.fft.fft(block)

        # Shift the STFT to center the frequency spectrum
        stft = np.fft.fftshift(stft)

        # Compute the stretched STFT
        start = int(len(stft) * (1 - stretch_factor) / 2)
        stop = int(len(stft) * (1 + stretch_factor) / 2)
        stretched_stft = np.zeros_like(stft)
        stretched_stft[start:stop] = stft[start:stop]

        # Shift the stretched STFT back to its original position
        stretched_stft = np.fft.ifftshift(stretched_stft)

        # Compute the inverse STFT of the stretched STFT
