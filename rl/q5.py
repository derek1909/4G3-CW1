import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation parameters
dt = 0.5
T_trial = 25.0
time_points = np.arange(0, T_trial+dt, dt)
n_steps = len(time_points)
N_trials = 1000
p = 0.5         # Partial reinforcement probability: 50% of trials deliver reward
gamma = 1.0
eps_boxcar = 0.01  # learning rate for boxcar representation

# Define stimulus: spike at 10s
def stimulus(t):
    return 1.0 if t==10.0 else 0.0

# Define reward: Gaussian around 20s with sigma=1, scaled by 1/2
def reward(t):
    return 0.5 * np.exp(-((t-20)**2)/(2))

# Build state representation for tapped delay line (length 25)
def build_state(history):
    """
    Build a tapped-delay-line-like state in reverse order.
    The newest time step will be at index 0,
    and the oldest time step will be at index 24.
    """
    length = len(history)
    # First ensure we only keep the last 25 steps if length exceeds 25
    if length <= 25:
        padded = np.concatenate([np.zeros(25 - length), history])
    else:
        padded = history[-25:]
    # Now reverse so the newest is at index 0
    return padded[::-1]


# Initialize weights
w = np.zeros(25)

# Lists to store data for the last 100 trials.
last_100_V = []
last_100_dV = []
last_100_delta = []
last_100_rewards = []  # Boolean flag: True if reward was delivered

for trial in range(N_trials):
    # Determine if reward is delivered this trial.
    reward_delivered = (np.random.rand() < p)
    
    stim_history = []
    V_trial = np.zeros(n_steps)
    dV_trial = np.zeros(n_steps)
    delta_trial = np.zeros(n_steps)
    
    state = build_state(np.array([]))
    V = 0.0
    r = 0.0
    
    for i, t in enumerate(time_points):
        prev_state = state
        prev_V = V
        prev_r = r

        current_stim = stimulus(t)
        stim_history.append(current_stim)
        state = build_state(np.array(stim_history))  # tapped delay state

        V = np.dot(w, np.cumsum(state))
        V_trial[i] = V

        if reward_delivered:
            r = reward(t)
        else:
            r = 0.0

        dV = gamma * V - prev_V
        delta = dV + prev_r

        dV_trial[i] = dV
        delta_trial[i] = delta

        w += eps_boxcar * delta * np.cumsum(prev_state)

    # Save data for the last 100 trials.
    if trial >= N_trials - 100:
        last_100_V.append(V_trial)
        last_100_dV.append(dV_trial)
        last_100_delta.append(delta_trial)
        last_100_rewards.append(reward_delivered)


last_100_V = np.array(last_100_V)         # Shape: (100, n_steps)
last_100_dV = np.array(last_100_dV)
last_100_delta = np.array(last_100_delta)
last_100_rewards = np.array(last_100_rewards)  # Shape: (100,)

# Create boolean indices for rewarded and unrewarded trials.
rewarded_trials = last_100_rewards.astype(bool)
unrewarded_trials = ~rewarded_trials

# Compute averages across trials.
V_rewarded_avg    = np.mean(last_100_V[rewarded_trials], axis=0)
V_unrewarded_avg  = np.mean(last_100_V[unrewarded_trials], axis=0)
V_overall_avg     = np.mean(last_100_V, axis=0)

dV_rewarded_avg   = np.mean(last_100_dV[rewarded_trials], axis=0)
dV_unrewarded_avg = np.mean(last_100_dV[unrewarded_trials], axis=0)
dV_overall_avg    = np.mean(last_100_dV, axis=0)

delta_rewarded_avg   = np.mean(last_100_delta[rewarded_trials], axis=0)
delta_unrewarded_avg = np.mean(last_100_delta[unrewarded_trials], axis=0)
delta_overall_avg    = np.mean(last_100_delta, axis=0)


fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

# Plot average Value V(t)
axs[0].plot(time_points, V_rewarded_avg, label='Rewarded', color='green')
axs[0].plot(time_points, V_unrewarded_avg, label='Unrewarded', color='red')
axs[0].plot(time_points, V_overall_avg, label='Overall', color='blue')
axs[0].set_title('Average Value V(t) for Last 100 Trials')
axs[0].set_ylabel('V(t)')
axs[0].legend()

# Plot average temporal difference dV(t)
axs[1].plot(time_points, dV_rewarded_avg, label='Rewarded', color='green')
axs[1].plot(time_points, dV_unrewarded_avg, label='Unrewarded', color='red')
axs[1].plot(time_points, dV_overall_avg, label='Overall', color='blue')
axs[1].set_title('Average Temporal Difference dV(t) for Last 100 Trials')
axs[1].set_ylabel('dV(t)')
axs[1].legend()

# Plot average TD error δ(t)
axs[2].plot(time_points, delta_rewarded_avg, label='Rewarded', color='green')
axs[2].plot(time_points, delta_unrewarded_avg, label='Unrewarded', color='red')
axs[2].plot(time_points, delta_overall_avg, label='Overall', color='blue')
axs[2].set_title('Average TD Error δ(t) for Last 100 Trials')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('δ(t)')
axs[2].legend()

plt.tight_layout()
plt.savefig('./rl/averages_last_100.png', dpi=300)
plt.close()


# Example piecewise function
def dopamine_activity(x, alpha=6, beta=6, x_star=0.27):
    """
    Computes the dopamine activity given x using
    the piecewise definition with parameters alpha, beta, and x_star.
    """
    if x < 0:
        return x / alpha
    elif x <= x_star:
        return x
    else:
        return x_star + (x - x_star) / beta

DA_values_last100 = np.zeros_like(last_100_delta)

for i in range(100):
    for t in range(last_100_delta.shape[1]):
        x_val = last_100_delta[i, t]
        DA_values_last100[i, t] = dopamine_activity(x_val, alpha=6, beta=6, x_star=0.27)

# Now average across the 100 trials for both x and DA
x_avg = np.mean(last_100_delta, axis=0)   # shape: (n_steps,)
DA_avg = np.mean(DA_values_last100, axis=0)  # shape: (n_steps,)

# Plot
plt.figure(figsize=(4,2.5))
plt.plot(time_points, x_avg, label='Average of TD error', color='blue')
plt.plot(time_points, DA_avg, label='Average of DA', color='orange')
plt.title('Dopamine Signal vs. TD Error (Last 100 Trials)')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.tight_layout()
plt.savefig('./rl/dopamine_plot.png', dpi=300)
plt.close()