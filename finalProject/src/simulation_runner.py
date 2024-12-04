from kalman_filter import run_kalman_filter
import numpy as np

# WIP, only calls the Kalman Filter code as of right now, edit to also call the particle filter code and run the two simulations
# Example parameters for Kalman Filter
A = np.array([[1]])  # State transition (position update)
B = np.array([[1]])  # Control input (velocity)
H = np.array([[1]])  # Observation matrix
Q = np.array([[0.1]])  # Process noise covariance
R = np.array([[1]])  # Measurement noise covariance
x0 = np.array([0])  # Initial state estimate
P0 = np.array([[1]])  # Initial covariance estimate
control_input = np.array([1])

# Generate ground truth and measurements
n_timesteps = 50
true_position = [0]
measurements = []
np.random.seed(42)

for t in range(1, n_timesteps):
    true_position.append(true_position[-1] + 1 + np.random.normal(0, 0.5))  # Velocity is 1
    measurements.append(true_position[-1] + np.random.normal(0, 1))

# Run the Kalman Filter using the imported function
estimates = run_kalman_filter(A, B, H, Q, R, x0, P0, measurements, control_input)

# Further analysis or visualization (e.g., comparing true positions, measurements, estimates)
 
