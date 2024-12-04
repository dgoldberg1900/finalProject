import numpy as np

# Define the Kalman Filter class
class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, u):
        # Predict the next state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Update the estimate via measurement
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.A.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self):
        return self.x

# Function to run the Kalman Filter (to be called by simulation_runner.py)
def run_kalman_filter(A, B, H, Q, R, x0, P0, measurements, control_input):
    kf = KalmanFilter(A, B, H, Q, R, x0, P0)
    estimates = []
    for z in measurements:
        kf.predict(control_input)
        kf.update(z)
        estimates.append(kf.get_state()[0])
    return estimates
 
