import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make("Taxi-v3")
n_actions = env.action_space.n
n_states = env.observation_space.n

Q = np.zeros((n_states, n_actions))

# Define hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

def epsilon_greedy(Q, state, epsilon):
    """ Selects an action using epsilon-greedy policy based on current Q-values """
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# Metrics for analysis
rewards_per_episode = []
steps_per_episode = []
q_value_changes = []

# Train the model over a number of episodes
num_episodes = 1000
previous_Q = np.copy(Q)

for episode in range(num_episodes):
    initial_state_info = env.reset()

    if isinstance(initial_state_info, tuple):
        state, _ = initial_state_info
    else:
        state = initial_state_info

    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done, _, info = env.step(action)

        # Update Q-table
        old_q_value = Q[state, action]
        best_future_value = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_future_value - Q[state, action])
        new_q_value = Q[state, action]

        state = next_state
        total_reward += reward
        steps += 1

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)  # Record steps taken in this episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Calculate average absolute change in Q-values
    q_value_changes.append(np.sum(np.abs(previous_Q - Q)) / (n_states * n_actions))
    previous_Q = np.copy(Q)

def plot_metrics(rewards, steps, q_value_changes):
    """ Plots learning performance, episode lengths, and Q-value changes over time """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(rewards)
    axs[0].set_title('Total Rewards per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(steps)
    axs[1].set_title('Steps per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')

    axs[2].plot(q_value_changes)
    axs[2].set_title('Average Q-value Change per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Q-value Change')

    plt.tight_layout()
    plt.show()

# Execute the plot function
plot_metrics(rewards_per_episode, steps_per_episode, q_value_changes)
