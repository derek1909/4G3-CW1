# for 3. tapped delay line.
import numpy as np
import matplotlib.pyplot as plt
import os

# Simulation parameters
dt = 0.5
T_trial = 25.0
time_points = np.arange(0, T_trial+dt, dt)
n_steps = len(time_points)
N_trials = 201
gamma = 1.0
eps_tapped = 0.2  # learning rate for tapped delay line
eps_boxcar = 0.01  # learning rate for boxcar representation

is_tapped = False

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

# Build state for boxcar: cumulative sum up to each lag
def build_state_boxcar(state):
    return 

# Initialize weights
w = np.zeros(25)

# Record variables for selected trials (every 10th trial)
selected_trials = []
V_rec = []      # value estimates
dV_rec = []     # temporal difference of value
delta_rec = []  # TD error

for trial in range(N_trials):
    stim_history = []
    V_trial = np.zeros(n_steps)
    dV_trial = np.zeros(n_steps)
    delta_trial = np.zeros(n_steps)

    state = build_state(np.array([]))
    V = 0
    r = 0

    # For each time step in trial
    for i, t in enumerate(time_points):
        # if t==0:
        #     continue
        prev_state = state
        prev_V = V
        prev_r = r

        # Update stimulus history
        current_stim = stimulus(t)
        stim_history.append(current_stim)
        state = build_state(np.array(stim_history)) # tapped delay state
        if is_tapped:
            V = np.dot(w, state)
        else:
            V = np.dot(w, np.cumsum(state))
        
        V_trial[i] = V

        # TD error
        r = reward(t)
        dV = gamma * V - prev_V
        delta = dV + prev_r

        dV_trial[i] = dV
        delta_trial[i] = delta

        # Update weights
        if is_tapped:
            w += eps_tapped * delta * prev_state
        else:
            w += eps_boxcar * delta * prev_state #np.cumsum(prev_state)

    # Store selected trial data
    if trial % 10 == 0:
        selected_trials.append(trial)
        V_rec.append(V_trial)
        dV_rec.append(dV_trial)
        delta_rec.append(delta_trial)

# 确保保存图片的文件夹存在
os.makedirs('./rl', exist_ok=True)

# # Plotting code: 绘制刺激和奖励随时间变化的图像
plt.figure(figsize=(8,2.5))
plt.plot(time_points, [reward(t) for t in time_points], 'k-', label='Reward')
plt.plot(time_points, [stimulus(t) for t in time_points], 'b--', label='Stimulus')
plt.xlabel('Time (s)')
plt.legend()
# plt.title('Stimulus and Reward Profile')
plt.savefig('./rl/stim_reward.png')  # 将图片保存到 ./rl 目录下
plt.close()  # 关闭图形窗口


# Plot V(t), dV(t), and delta(t) for the selected 21 trials in three subplots
# Create a transparency gradient: same fixed color but alpha changes linearly.
alphas = np.linspace(0.2, 1, len(selected_trials))
fixed_color = 'blue'

# Create a figure with 3 subplots sharing the same x-axis
fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

# Loop over each selected trial and plot using the corresponding alpha value.
for i, trial_idx in enumerate(selected_trials):
    axs[0].plot(time_points, V_rec[i], color=fixed_color, alpha=alphas[i], label=f'Trial {trial_idx}')
    axs[1].plot(time_points, dV_rec[i], color=fixed_color, alpha=alphas[i], label=f'Trial {trial_idx}')
    axs[2].plot(time_points, delta_rec[i], color=fixed_color, alpha=alphas[i], label=f'Trial {trial_idx}')

# Set titles and axis labels
axs[0].set_title('Value V(t)')
axs[0].set_ylabel('V(t)')

axs[1].set_title('Temporal Difference of Value dV(t)')
axs[1].set_ylabel('dV(t)')

axs[2].set_title('TD Error δ(t)')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('δ(t)')

plt.tight_layout()

# Ensure the output directory exists
os.makedirs('./rl', exist_ok=True)

# Save the figure with high resolution (dpi = 300) and then close the figure.
if is_tapped:
    plt.savefig('./rl/tapped_plot.png', dpi=300)
else:    
    plt.savefig('./rl/boxcar_plot.png', dpi=300)
plt.close()

# 类似地，可绘制每个 selected trial 的 V, dV 和 delta 的三联图

if __name__ == '__main__':
    pass