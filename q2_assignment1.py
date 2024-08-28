import numpy as np
import matplotlib.pyplot as plt

# Common Functions
def add_action(state, action):
    if action == 0:
        return [state[0] - 1, state[1]]  # Up
    elif action == 1:
        return [state[0] + 1, state[1]]  # Down
    elif action == 2:
        return [state[0], state[1] + 1]  # Right
    elif action == 3:
        return [state[0], state[1] - 1]  # Left
    return state

def accessible(state, movability):
    return 0 <= state[0] < 9 and 0 <= state[1] < 9 and movability[state[0]][state[1]] == 1

def get_new_state(state, action, movability):
    if state == [2, 2]:  # Tunnel entrance
        return [6, 6]  # Tunnel exit
    new_state = add_action(state, action)
    if not accessible(new_state, movability):
        return state  # Stay in place if moving into a wall or outside grid
    return new_state

def get_reward(state, next_state):
    if state == next_state and (next_state[0] < 0 or next_state[0] >= 9 or next_state[1] < 0 or next_state[1] >= 9):
        return -1.0  # Penalty for trying to move outside the grid
    return 1.0 if next_state == [8, 8] else 0.0

def plot_policy(policy, blockages, tunnel_in, tunnel_out, title):
    grid_size = 9
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if policy[i, j] == 0:  # Up
                U[i, j] = -1
            elif policy[i, j] == 1:  # Down
                U[i, j] = 1
            elif policy[i, j] == 2:  # Right
                V[i, j] = 1
            elif policy[i, j] == 3:  # Left
                V[i, j] = -1

    plt.figure(figsize=(8, 8))
    plt.quiver(Y, X, V, U)
    
    for (bx, by) in blockages:
        plt.gca().add_patch(plt.Rectangle((by-0.5, bx-0.5), 1, 1, fill=True, color="black"))
    
    plt.gca().add_patch(plt.Rectangle((tunnel_in[1]-0.5, tunnel_in[0]-0.5), 1, 1, fill=True, color="red", alpha=0.6))
    plt.gca().add_patch(plt.Rectangle((tunnel_out[1]-0.5, tunnel_out[0]-0.5), 1, 1, fill=True, color="blue", alpha=0.6))
    
    plt.title(title)
    plt.show()

# Value Iteration
def value_iteration(states, movability, gamma=0.9, theta=1e-6):
    values = {tuple(state): 0.0 for state in states}
    values[(8, 8)] = 1.0  # Terminal state

    while True:
        delta = 0
        for state in states:
            if state == [8, 8]:
                continue  # Skip terminal state
            v = values[tuple(state)]
            max_value = float('-inf')
            for action in [0, 1, 2, 3]:
                new_state = get_new_state(state, action, movability)
                reward = get_reward(state, new_state)
                value = reward + gamma * values[tuple(new_state)]
                max_value = max(max_value, value)

            values[tuple(state)] = max_value
            delta = max(delta, abs(v - max_value))
        if delta < theta:
            break
    return values

def extract_policy(states, values, movability, gamma=0.9):
    policy = np.full((9, 9), -1)  # Store best action for each state

    for i in range(9):
        for j in range(9):
            if [i, j] == [8, 8]:
                continue  # Goal state
            elif movability[i][j] == 0:
                continue  # Blockage
            max_value = float('-inf')
            best_action = -1
            for action in [0, 1, 2, 3]:
                new_state = get_new_state([i, j], action, movability)
                reward = get_reward([i, j], new_state)
                value = reward + gamma * values[tuple(new_state)]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[i, j] = best_action
    return policy

# Policy Iteration
def policy_evaluation(policy, states, movability, gamma=0.9, theta=1e-6):
    values = {tuple(state): 0.0 for state in states}
    values[(8, 8)] = 1.0  # Terminal state

    while True:
        delta = 0
        for state in states:
            if state == [8, 8]:
                continue  # Skip terminal state
            v = values[tuple(state)]
            action = policy[state[0], state[1]]
            new_state = get_new_state(state, action, movability)
            reward = get_reward(state, new_state)
            values[tuple(state)] = reward + gamma * values[tuple(new_state)]
            delta = max(delta, abs(v - values[tuple(state)]))
        if delta < theta:
            break
    return values

def policy_improvement(states, values, movability, gamma=0.9):
    policy = np.full((9, 9), -1)

    for i in range(9):
        for j in range(9):
            if [i, j] == [8, 8]:
                continue  # Goal state
            elif movability[i][j] == 0:
                continue  # Blockage
            max_value = float('-inf')
            best_action = -1
            for action in [0, 1, 2, 3]:
                new_state = get_new_state([i, j], action, movability)
                reward = get_reward([i, j], new_state)
                value = reward + gamma * values[tuple(new_state)]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[i, j] = best_action
    return policy

def policy_iteration(states, movability, gamma=0.9, theta=1e-6):
    policy = np.random.choice([0, 1, 2, 3], size=(9, 9))
    policy[8, 8] = -1  # Goal state has no action

    while True:
        values = policy_evaluation(policy, states, movability, gamma, theta)
        new_policy = policy_improvement(states, values, movability, gamma)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    return policy

# Main Function
def main():
    movability = np.ones((9, 9), dtype=int)

    blockages = [
        (1, 3), (2, 3), (3, 3), (3, 1), (3, 2),
        (5, 8), (5, 7), (5, 6), (5, 5), (6, 5), (7, 5), (8, 5)
    ]
    for blockage in blockages:
        movability[blockage[0]][blockage[1]] = 0

    states = [[i, j] for i in range(9) for j in range(9)]

    # Value Iteration
    values = value_iteration(states, movability)
    policy_vi = extract_policy(states, values, movability)
    plot_policy(policy_vi, blockages, [2, 2], [6, 6], "Optimal Policy - Value Iteration")

    # Policy Iteration
    policy_pi = policy_iteration(states, movability)
    plot_policy(policy_pi, blockages, [2, 2], [6, 6], "Optimal Policy - Policy Iteration")

if __name__ == "__main__":
    main()
