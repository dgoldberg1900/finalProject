# Progress Report v2: Milestone 2 Updates

## Status:

1. **Research phase (FINISHED)**
   - Research into the main localization problems of robotic navigation.
   - Research into existing Kalman and Particle filtering solutions including what they are, how they work, and how they are implemented.
2. **Algorithmic Design and Implementation (IN PROGRESS)**
3. **Research paper writing (IN PROGRESS)**

## Algorithmic Design and Implementation
The algorithmic design and implementation phase is underway with Daniel covering the Kalman implementation and Andrew covering the Particle Filtering implementation.

So far both the `kalman_filter.py` and the `particle_filter.py` have had major updates with important methods and classes added. Both filters require a few more additions before the simulation_runner can be established and initial tests conducted.


## Particle Filter Localization Algorithm

The algorithm design demonstrates the system's state as a set of particles each with a weight that indicates the liklihood based on the current observation. The algorithm iterately refines an estimate of the state of the system using a motion model to predict particle states. This is down by updating the weights using an abstract measurement model and through resampling particles based on their weights to focus on high-probability regions of the state space.

### Inputs:
- **M**: Number of particles
- **X**: Initial particle states
- **Z_t**: Sensor measurements at time `t`
- **U_t**: Control inputs at time `t`
- **MotionModel(U_t, X)**: Predicts state based on control inputs
- **MeasurementModel(Z_t, X)**: Likelihood of sensor observation given state

### Outputs:
- **X_t**: Updated particle set representing the estimated pose distribution

---

```python
# Initialize particles
Initialize particles {X_0^(1), X_0^(2), ..., X_0^(M)} randomly across the state space
Assign equal weights w_0^(i) = 1 / M to each particle

# Time Step Loop
For each timestep `t`:
    1. Prediction (Motion Update):
       For each particle `i`:
           X_t^(i) = MotionModel(U_t, X_t-1^(i)) + Noise
           
    2. Update (Measurement Likelihood):
       For each particle `i`:
           w_t^(i) = MeasurementModel(Z_t, X_t^(i))
           
    3. Normalization:
       Normalize weights: w_t^(i) = w_t^(i) / Σ w_t^(j) for all `j`

    4. Resampling:
       Resample `M` particles based on weights w_t^(i) using a sampling method 
       Reset weights w_t^(i) = 1 / M for all `i`

    5. Estimate State:
       Compute estimated state as the weighted mean of particles:
       X_estimate_t = Σ (w_t^(i) * X_t^(i)) for all `i`

    Return updated particle set X_t
```

## Research Paper 

The research paper has been templated and the draft is currently being written in parallel with our code design and implementation. Our paper is hosted on Overleaf which allows for seamless collaboration using LaTeX. The current status of the paper is a little rough with basic introductory information that is set to be edited. Our plan is to revise the paper gradually with a greater importance place on algorithm implementation for the time being.

*** Research Paper Link ***
[Final Paper Draft](https://www.overleaf.com/read/kqxkqjnqqprs#1c0929)
