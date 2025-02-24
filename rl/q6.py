import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.5
T_trial = 25.0
time_points = np.arange(0, T_trial + dt, dt)
n_steps = len(time_points)
N_trials = 1000
gamma = 1.0
eps_boxcar = 0.01  # Learning rate for the boxcar representation

# Define stimulus: spike at 10s
def stimulus(t):
    return 1.0 if np.isclose(t, 10.0) else 0.0

# Define reward: Gaussian around 20s with sigma=1, scaled by 1/2
def reward(t):
    return 0.5 * np.exp(-((t - 20)**2) / 2)

# Build state representation for a tapped delay line (length = 25)
def build_state(history):
    length = len(history)
    if length <= 25:
        padded = np.concatenate([np.zeros(25 - length), history])
    else:
        padded = history[-25:]
    return padded[::-1]  # Reverse: newest at index 0

# Dopamine activity function (piecewise nonlinearity)
def dopamine_activity(x, alpha=6, beta=6, x_star=0.27):
    if x < 0:
        return x / alpha
    elif x <= x_star:
        return x
    else:
        return x_star + (x - x_star) / beta

# List of reward probabilities to test
p_list = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = {0.0: 'blue', 0.25: 'green', 0.5: 'orange', 0.75: 'red', 1.0: 'purple'}

# Dictionaries to store average dopamine and delta curves for each probability
avg_DA_curves = {}
avg_delta_curves = {}

# Loop over each reward probability
for p in p_list:
    # Initialize weights for each independent experiment
    w = np.zeros(25)
    
    # Lists to store delta and dopamine time courses for the last 100 trials
    last_100_delta = []
    last_100_da = []
    
    for trial in range(N_trials):
        # Determine whether a reward is delivered on this trial.
        reward_delivered = (np.random.rand() < p)
        
        stim_history = []
        delta_trial = np.zeros(n_steps)
        
        state = build_state(np.array([]))
        V = 0.0
        r = 0.0
        
        # Loop through time points of the trial
        for i, t in enumerate(time_points):
            prev_state = state
            prev_V = V
            prev_r = r
            
            current_stim = stimulus(t)
            stim_history.append(current_stim)
            state = build_state(np.array(stim_history))
            
            V = np.dot(w, np.cumsum(state))
            
            # Apply reward if delivered on this trial
            r = reward(t) if reward_delivered else 0.0
            
            dV = gamma * V - prev_V
            delta = dV + prev_r
            delta_trial[i] = delta
            
            # Update weights using the boxcar representation
            w += eps_boxcar * delta * np.cumsum(prev_state)
        
        # For each trial, compute the dopamine signal from delta_trial
        da_trial = np.array([dopamine_activity(x) for x in delta_trial])
        
        # Save trial data if it's one of the last 100 trials
        if trial >= N_trials - 100:
            last_100_delta.append(delta_trial)
            last_100_da.append(da_trial)
    
    # Convert lists to numpy arrays and compute the average over the last 100 trials
    last_100_delta = np.array(last_100_delta)
    last_100_da = np.array(last_100_da)
    
    avg_delta = np.mean(last_100_delta, axis=0)
    avg_da = np.mean(last_100_da, axis=0)
    
    # Store the averaged curves for plotting
    avg_delta_curves[p] = avg_delta
    avg_DA_curves[p] = avg_da

# Plotting: create a figure with two subplots for average DA and average delta
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot average dopamine time courses for each reward probability
for p in p_list:
    axs[0].plot(time_points, avg_DA_curves[p], label=f'p = {p}', color=colors[p])
axs[0].set_ylabel('Average Dopamine Signal')
axs[0].set_title('Average Dopamine (Last 100 Trials)')
axs[0].legend()

# Plot average TD error (delta) time courses for each reward probability
for p in p_list:
    axs[1].plot(time_points, avg_delta_curves[p], label=f'p = {p}', color=colors[p])
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Average TD Error (delta)')
axs[1].set_title('Average TD Error (Last 100 Trials)')
axs[1].legend()

plt.tight_layout()

# Save the combined figure to the 'rl' folder (make sure this folder exists)
plt.savefig('./rl/avg_dopamine_and_delta_by_p.png', dpi=300)
plt.close()