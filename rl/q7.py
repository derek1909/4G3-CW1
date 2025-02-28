import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters for simulation
dt = 0.5
T_trial = 25.0
time_points = np.arange(0, T_trial+dt, dt)  # 0, 0.5, ..., 25.0
n_steps = len(time_points)
N_trials = 10000
gamma = 1.0
eps_boxcar = 0.01  # learning rate for boxcar representation

# Define a list of reward probabilities to simulate.
p_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9,1.0]

# Define stimulus: spike at 10s.
def stimulus(t):
    return 1.0 if t == 10.0 else 0.0

# Define reward: Gaussian around 20s with sigma=1, scaled by 1/2.
def reward(t):
    return 0.5 * np.exp(-((t-20)**2)/(2))

# Build tapped-delay-line state (in reverse order).
def build_state(history):
    length = len(history)
    if length <= 25:
        padded = np.concatenate([np.zeros(25 - length), history])
    else:
        padded = history[-25:]
    return padded[::-1]

# Build boxcar state by taking cumulative sum.
def build_state_boxcar(state):
    return np.cumsum(state)

# Define the dopamine function as given:
# DA(x) = x/alpha if x<0; = x if 0<= x < x*; = x* + (x-x*)/beta if x >= x*
def DA(x, alpha=6.0, beta=6.0, x_star=0.27):
    x = np.array(x)
    da = np.zeros_like(x)
    # For negative x:
    mask_neg = (x < 0)
    da[mask_neg] = x[mask_neg] / alpha
    # For x between 0 and x_star:
    mask_mid = (x >= 0) & (x < x_star)
    da[mask_mid] = x[mask_mid]
    # For x >= x_star:
    mask_high = (x >= x_star)
    da[mask_high] = x_star + (x[mask_high] - x_star)/beta
    return da

# Define a window half-width (in seconds) to search for the maximum dopamine value
window_half = 1.0

# Create boolean masks for the stimulus and reward windows
stim_window = (time_points >= (10.0 - window_half)) & (time_points <= (10.0 + window_half))
reward_window = (time_points >= (20.0 - window_half)) & (time_points <= (20.0 + window_half))

# Initialize arrays to hold the dopamine peaks (averaged over trials).
avg_DA_stim = []
avg_DA_reward = []

# Loop over each reward probability p.
for p in p_values:
    # Initialize weight vector for the boxcar representation.
    w = np.zeros(25)
    # Lists to store dopamine peaks for each trial.
    DA_stim_trials = []
    DA_reward_trials = []
    
    for trial in range(N_trials):
        # Determine if reward is delivered for this trial.
        reward_delivered = (np.random.rand() < p)
        
        stim_history = []
        V_trial = np.zeros(n_steps)
        delta_trial = np.zeros(n_steps)
        
        # Initialize state, value, and reward.
        state = build_state(np.array([]))
        V = 0.0
        r = 0.0
        
        for i, t in enumerate(time_points):
            # Update stimulus history and rebuild state.
            current_stim = stimulus(t)
            stim_history.append(current_stim)
            state = build_state(np.array(stim_history))
            
            # Compute current value estimate.
            V = np.dot(w, np.cumsum(state))
            V_trial[i] = V
            
            # Compute reward: if reinforced, use reward(t); otherwise, 0.
            if reward_delivered:
                r = reward(t)
            else:
                r = 0.0
            
            # Compute TD error: δ = r + γ * V_next - V.
            if i < n_steps - 1:
                next_stim = stimulus(time_points[i+1])
                next_history = stim_history + [next_stim]
                next_state = build_state(np.array(next_history))
                V_next = np.dot(w, np.cumsum(next_state))
            else:
                V_next = 0.0
            
            delta = r + gamma * V_next - V
            delta_trial[i] = delta
            
            # Update weights using the current boxcar state.
            w += eps_boxcar * delta * np.cumsum(state)
        
        # After each trial, compute the dopamine signal from the TD errors.
        DA_signal = DA(delta_trial)
        
        # Instead of using a fixed index, find the maximum dopamine in the stimulus and reward windows.
        DA_stim_trials.append(np.max(DA_signal[stim_window]))
        DA_reward_trials.append(np.max(DA_signal[reward_window]))
    
    # Average over all trials (or a subset, e.g., last 100) for this p.
    avg_DA_stim.append(np.mean(DA_stim_trials))
    avg_DA_reward.append(np.mean(DA_reward_trials))

# Plot the average dopamine at the stimulus and reward windows as a function of p.
plt.figure(figsize=(6,3))
plt.plot(p_values, avg_DA_stim, marker='o', linestyle='-', label='Dopamine at Stimulus (max in window)')
plt.plot(p_values, avg_DA_reward, marker='s', linestyle='-', label='Dopamine at Reward (max in window)')
plt.xlabel('Reward Probability (p)')
plt.ylabel('Average Dopamine Level')
plt.title('Average Dopamine Level vs. Reward Probability')
plt.legend()
plt.grid(True)
os.makedirs('./rl', exist_ok=True)
plt.savefig('./rl/DA_vs_p.png', dpi=300)
plt.close()

# Print the results for inspection.
print("Reward Probability (p):", p_values)
print("Avg DA at Stimulus:", avg_DA_stim)
print("Avg DA at Reward:", avg_DA_reward)