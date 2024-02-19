import numpy as np


def test_roundtrip_fft_ift():
    # Generate a test signal
    audio = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1, 44100))

    # Add white Gaussian noise to the signal
    noise_amplitude = 0.1
    noise = noise_amplitude * np.random.randn(len(audio))
    audio += noise

    # Compute the FFT of the test signal
    fft = np.fft.fft(audio)

    # Compute the IFFT of the FFT
    ifft = np.fft.ifft(fft)

    # Compute the relative error between the original and reconstructed signals
    relative_error = np.abs(audio - ifft) / np.abs(audio)

    # Compute the RMS error between the original and reconstructed signals
    rms_error = np.sqrt(np.mean((audio - ifft) ** 2))

    # Compute the PSNR between the original and reconstructed signals
    psnr = 10 * np.log10(np.max(audio) ** 2 / np.mean((audio - ifft) ** 2))

    # Compute the SNR between the original and reconstructed signals
    snr = np.mean(audio ** 2) / np.mean((audio - ifft) ** 2)

    # Print the results
    # print("Relative error:", relative_error)
    print("RMS error:", np.abs(rms_error))
    print("PSNR:", psnr)
    print("SNR:", snr)


test_roundtrip_fft_ift()
