import numpy as np


class ParticleFilter:
    """
    Particle Filter class.

    Attributes:
    - num_particles: Number of particles
    - state_dim: Dimension of the state space
    - motion_model: Function to predict particle states
    - measurement_model: Function to compute measurement likelihoods
    - particles: Particle states: num_particles x state_dim array
    - weights: Particle weights

    Methods:
    - predict: Predict the new state of the particles based on the motion model and control input
    - update: Update the particle weights based on the measurement model
    - resample: Resample particles based on their weights to focus on high-probability regions
    - estimate: Compute the weighted mean of particles as the estimated state
    """

    def __init__(self, num_particles, state_dim, motion_model, measurement_model, init_particles, init_weights):
        self.num_particles = num_particles  # Number of particles
        self.state_dim = state_dim  # Dimension of the state space
        self.motion_model = motion_model  # Function to predict particle states
        self.measurement_model = measurement_model  # Function to compute measurement likelihoods
        self.particles = init_particles  # Initial particle states: num_particles x state_dim array
        self.weights = init_weights  # Initial weights 
    
    def predict(self, control_input):
        """
        Predict the new state of the particles based on the motion model and control input.
        """
        self.particles = np.array([self.motion_model(p, control_input) for p in self.particles])

    def update(self, measurement):
        """
        Update the particle weights based on the measurement model.
        """
        self.weights = np.array([self.measurement_model(measurement, p) for p in self.particles])
        self.weights /= np.sum(self.weights)  # Normalize weights

    def resample(self):
        """
        Resample particles based on their weights to focus on high-probability regions.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

        # Reset weights to uniform distribution after resampling to avoid degeneracy
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """
        Compute the weighted mean of particles as the estimated state.
        """
        return np.average(self.particles, weights=self.weights, axis=0)


def motion_model(state, control_input):
    # Additive Gaussian noise
    noise = np.random.normal(0, 1, size=state.shape) 
    return state + control_input + noise

def measurement_model(measurement, state):
    distance = np.linalg.norm(measurement - state) 
    # Gaussian likelihood with unit variance
    likelihood = np.exp(-0.5 * (distance ** 2)) 
    return likelihood

# Function to run the Particle Filter (to be called by a simulation runner)
def run_particle_filter(num_particles, state_dim, motion_model, measurement_model, init_particles, init_weights, measurements, control_inputs):
    pf = ParticleFilter(num_particles, state_dim, motion_model, measurement_model, init_particles, init_weights)
    estimates = []
    for t, measurement in enumerate(measurements):
        control_input = control_inputs[t]
        pf.predict(control_input)
        pf.update(measurement)
        pf.resample()
        estimates.append(pf.estimate())
    return estimates
