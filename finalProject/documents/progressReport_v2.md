# Progress Report v2: Milestone 2 Updates

## Status:

1. **Research phase (FINISHED)**
   - Research into the main localization problems of robotic navigation.
   - Research into existing Kalman and Particle filtering solutions including what they are, how they work, and how they are implemented.
2. **Algorithmic Design and Implementation (IN PROGRESS)**
3. **Research paper writing (IN PROGRESS)**

## Algorithmic Design and Implementation
The algorithmic design and implementation phase is underway with Daniel covering the Kalman implementation and Andrew covering the Particle Filtering implementation. So far we have:


## Particle Filter Localization Algorithm

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
