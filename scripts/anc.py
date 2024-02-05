import numpy as np
import matplotlib.pyplot as plt


def snr(output1, output2, reference):
    return np.mean(reference ** 2) / np.mean((output1 - output2) ** 2)

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


def anc_adaptive_filter(audio):
    # Initialize the adaptive filter
    filter_length = 31
    weights = np.zeros(filter_length)

    # Run the adaptive filter
    output = np.zeros_like(audio)
    input_history = np.zeros_like(weights)
    for i in range(len(audio)):
        # Shift the input history and append the current input sample
        input_history = np.roll(input_history, -1)
        input_history[-1] = audio[i]

        # Compute the filter output
        output[i] = weights.dot(input_history)

        # Update the filter weights
        error = audio[i] - output[i]
        weights += 0.1 * error * input_history

    # Subtract the filtered noise from the original signal to obtain the noise-cancelled signal
    # noise_cancelled = audio - output
    return output


def anc_kalman_filter(signal):
    # Define the system state
    state = np.zeros((2, 1))  # initial state (position and velocity)

    # Define the system matrix
    A = np.array([[1, 1], [0, 1]])  # state transition matrix

    # Define the process noise covariance
    Q = np.array([[0.001, 0.001], [0.001, 0.01]])  # covariance matrix

    # Define the measurement matrix
    H = np.array([[1, 0]])  # measurement matrix

    # Define the measurement noise covariance
    R = np.array([[0.1]])  # covariance matrix

    # Define the Kalman gain
    K = np.zeros((2, 1))  # initial Kalman gain

    # Define the estimation error covariance
    P = np.array([[1, 0], [0, 1]])  # initial estimation error covariance

    # Initialize the output signal
    output = np.zeros_like(signal)

    # Iterate over the time steps
    for i in range(1, len(signal)):
        # Predict the state
        state_pred = A @ state  # matrix multiplication
        P_pred = A @ P @ A.T + Q  # covariance prediction

        # Get the measurement
        measurement = signal[i]

        # Update the Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)

        # Update the state estimate
        state = state_pred + K @ (measurement - H @ state_pred)

        # Update the estimation error covariance
        P = (np.eye(2) - K @ H) @ P_pred

        # Store the output sample
        output[i] = H @ state


def gen_signal_with_noise(sample_rate, duration):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)*0.5
    signal += np.sin(2 * np.pi * 220 * t)*0.5
    signal += np.sin(2 * np.pi * 110 * t)*0.25
    noise = np.random.normal(scale=0.25, size=len(t))
    return (signal, noise, t)


# Generate a synthetic audio signal with noise
sample_rate = 22050
duration = 0.25

signal, noise, t = gen_signal_with_noise(sample_rate, duration)
noisy_signal = signal+noise*1.1
output = anc_adaptive_filter(noisy_signal)

# print(f'SNR: {snr(signal,noise,signal)}')
print(f'SNR: {snr(noisy_signal,signal,noise)}')
print(f'SNR: {snr(output,signal,noise)}')
# Plot the input and output signals
plt.plot(t, noisy_signal, label='Noisy')
plt.plot(t, output, label='Clean')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Plot the spectrum of the signal
frequencies = np.fft.rfftfreq(len(noisy_signal), 1.0 / sample_rate)
plt.semilogx(frequencies, np.abs(np.fft.rfft(normalize_2d(noisy_signal))) / sample_rate,label='Noisy')
plt.semilogx(frequencies, np.abs(np.fft.rfft(normalize_2d(output))) / sample_rate,label='Clean')
plt.gca().set_yscale('log')
plt.xlabel("Normalized frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.show()
