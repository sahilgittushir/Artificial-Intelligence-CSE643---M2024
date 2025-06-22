import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba
import csv


# ### Setup Environment

def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT


def setup_environment(seed=123):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)
    return GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS


# ### Sensor Movements

def simulate_roomba(T, movement_policy, sigma):
    """Simulate the movement of a Roomba robot for T time steps and generate noisy observations."""
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS, is_obstacle, start_pos, start_heading, movement_policy)

    true_positions = []
    observations = []
    headings = []

    print(f"Simulating Roomba movement for policy: {movement_policy}")
    for _ in tqdm(range(T), desc="Simulating Movement"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations


# ### Hidden Markov Model Components

def generate_states(grid_width, grid_height, headings):
    """Generate all possible states for the HMM."""
    states = []
    for x in range(grid_width):
        for y in range(grid_height):
            for h in headings:
                states.append(((x, y), h))
    return states


def emission_probability(state, observation, sigma):
    """Calculate the emission probability in log form for a given state and observation using a Gaussian distribution."""
    position, _ = state  # Ignore the heading
    obs_x, obs_y = observation
    true_x, true_y = position
    log_prob = -((obs_x - true_x) ** 2 + (obs_y - true_y) ** 2) / (2 * sigma ** 2)
    return log_prob


def transition_probability(prev_state, curr_state, movement_policy):
    """Calculate the transition probability in log form between two states based on a given movement policy."""
    prev_position, prev_heading = prev_state
    curr_position, curr_heading = curr_state

    if movement_policy == 'random_walk':
        return np.log(1 / len(HEADINGS)) if is_valid_transition(prev_position, curr_position) else -np.inf

    if movement_policy == 'straight_until_obstacle':
        if curr_position == next_position(prev_position, prev_heading):
            return 0.0
        if is_valid_transition(prev_position, curr_position):
            return np.log(1 / (len(HEADINGS) - 1))
    return -np.inf


def is_valid_transition(prev_position, curr_position):
    """Check if the transition between two positions is valid."""
    x, y = curr_position
    return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and abs(x - prev_position[0]) <= 1 and abs(y - prev_position[1]) <= 1


def next_position(position, heading):
    """Calculate the next position based on the current heading."""
    dx, dy = MOVEMENTS[heading]
    return position[0] + dx, position[1] + dy


# ### Viterbi Algorithm

def viterbi(observations, start_state, movement_policy, states, sigma):
    """Perform the Viterbi algorithm to find the most likely sequence of states given a series of observations."""
    T = len(observations)
    V = [{} for _ in range(T)]
    backpointer = [{} for _ in range(T)]

    for state in states:
        V[0][state] = emission_probability(state, observations[0], sigma)
        backpointer[0][state] = None

    for t in range(1, T):
        for curr_state in states:
            max_prob, prev_state = max(
                (V[t - 1][prev_state] + transition_probability(prev_state, curr_state, movement_policy), prev_state)
                for prev_state in states
            )
            V[t][curr_state] = max_prob + emission_probability(curr_state, observations[t], sigma)
            backpointer[t][curr_state] = prev_state

    max_prob, last_state = max((V[T - 1][state], state) for state in states)

    path = []
    for t in range(T - 1, -1, -1):
        path.insert(0, last_state)
        last_state = backpointer[t][last_state]

    return path


# ### Evaluation Functions

def getestimatedPath(policy, results, states, sigma):
    """Estimate the path of the Roomba using the Viterbi algorithm for a specified policy."""
    print(f"\nProcessing policy: {policy}")
    data = results[policy]
    observations = data['observations']
    start_state = (data['true_positions'][0], data['headings'][0])
    estimated_path = viterbi(observations, start_state, policy, states, sigma)
    return data['true_positions'], estimated_path


def evaluate_viterbi(estimated_path, true_positions, T, policy):
    """Evaluate the accuracy of the Viterbi algorithm's estimated path compared to the true path."""
    correct = 0
    for true_pos, est_state in zip(true_positions, estimated_path):
        if true_pos == est_state[0]:
            correct += 1
    accuracy = correct / T * 100
    print(f"Tracking accuracy for {policy.replace('_', ' ')} policy: {accuracy:.2f}%")


def plot_results(true_positions, observations, estimated_path, policy, seed):
    """Plot the true and estimated paths of the Roomba along with the noisy observations."""
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    obs_x = [obs[0] for obs in observations]
    obs_y = [obs[1] for obs in observations]
    est_x = [state[0][0] for state in estimated_path]
    est_y = [state[0][1] for state in estimated_path]

    plt.figure(figsize=(12, 6))

    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=15, label='Observations')
    plt.plot(est_x, est_y, 'b--', label='Estimated Path', linewidth=2)
    plt.title(f"{policy.title()} Policy - Seed {seed}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)

    fname = f"{policy}_Policy_Seed_{seed}_Path.png"
    plt.savefig(fname)
    print(f"Saved plot: {fname}")


# ### Main Execution

if __name__ == "__main__":
    seed_values = [123, 456, 789]  # Chosen for varied scenarios
    policies = ['random_walk', 'straight_until_obstacle']
    sigma = 1.0
    T = 50
    estimated_paths = []

    for seed in seed_values:
        setup_environment(seed)
        states = generate_states(GRID_WIDTH, GRID_HEIGHT, HEADINGS)

        results = {}
        for policy in policies:
            true_positions, headings, observations = simulate_roomba(T, policy, sigma)
            results[policy] = {'true_positions': true_positions, 'headings': headings, 'observations': observations}

        for policy in policies:
            true_positions, estimated_path = getestimatedPath(policy, results, states, sigma)
            evaluate_viterbi(estimated_path, true_positions, T, policy)
            plot_results(true_positions, results[policy]['observations'], estimated_path, policy, seed)
            estimated_paths.append([seed, policy, estimated_path])

    with open("estimated_paths.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Seed Value", "Policy Name", "Estimated Path"])
        for row in estimated_paths:
            writer.writerow(row)
