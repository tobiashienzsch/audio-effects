import pytest
import numpy as np
# import numpy as np


def stretch(audio, stretch_factor, block_size, window):
    # Pad the audio signal with zeros if its length is not divisible by the block size
    padding = block_size - len(audio) % block_size
    audio = np.pad(audio, (0, padding), 'constant')

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
        stretched_block = np.fft.ifft(stretched_stft)

        # Overlap-add the stretched block onto the output array
        stretched_audio[i:i+block_size] += np.real(stretched_block)

    # Trim the padded zeros from the output
    stretched_audio = stretched_audio[:len(audio)-padding]

    # Return the stretched audio
    return stretched_audio

# from stretch_function import stretch

# from .stretch import stretch


def test_stretch():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Stretch the audio signal by a factor of 2
    stretch_factor = 2.0
    stretched_audio = stretch(audio, stretch_factor, 2048, np.hanning(2048))

    # Check that the length of the stretched audio is correct
    print(len(stretched_audio))
    print(len(audio) * stretch_factor)
    assert len(stretched_audio) == len(audio) * stretch_factor


def test_stretch_minimum_block_size():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Stretch the audio signal by a factor of 2
    stretch_factor = 2.0
    stretched_audio = stretch(audio, stretch_factor, 1, np.hanning(1))

    # Check that the length of the stretched audio is correct
    assert len(stretched_audio) == len(audio) * stretch_factor


def test_stretch_maximum_block_size():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Stretch the audio signal by a factor of 2
    stretch_factor = 2.0
    stretched_audio = stretch(audio, stretch_factor,
                              len(audio), np.hanning(len(audio)))

    # Check that the length of the stretched audio is correct
    assert len(stretched_audio) == len(audio) * stretch_factor


def test_stretch_zero_stretch_factor():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Stretch the audio signal by a factor of 0
    stretch_factor = 0.0
    stretched_audio = stretch(audio, stretch_factor, 2048, np.hanning(2048))

    # Check that the length of the stretched audio is correct
    assert len(stretched_audio) == len(audio) * stretch_factor


def test_stretch_invalid_stretch_factor():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Stretch the audio signal by a negative factor
    stretch_factor = -1.0
    with pytest.raises(ValueError):
        stretched_audio = stretch(
            audio, stretch_factor, 2048, np.hanning(2048))


test_stretch()
test_stretch_minimum_block_size()
test_stretch_maximum_block_size()
test_stretch_zero_stretch_factor()
test_stretch_invalid_stretch_factor()
