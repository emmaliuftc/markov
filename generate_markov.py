import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate a particle in a double-well potential; basically goes towards -1 or 1

np.random.seed(42)
n_steps = 20000
dt = 0.01 
noise_scalar = 0.3

trajectory = np.zeros(n_steps)
x = -1.0 # Start in the left well

for t in range(1, n_steps):
    force = -4 * (x**3) + 4 * x
    # Update poistion - deterministic force + some random noise
    x += force * dt + np.random.normal(0,noise_scalar)
    trajectory[t] = x

X = trajectory.reshape(-1, 1)

# Find states

n_states = 2
kmeans = KMeans(n_clusters=n_states, random_state=42, n_init="auto")
discrete_trajectory = kmeans.fit_predict(X)

# Compute transition matrix

tau = 10 # the transition/lag time in the matrix
count_matrix = np.zeros((n_states,n_states))

for i in range(len(discrete_trajectory) - tau):
    initial_state = discrete_trajectory[i] # Tracking the intial state
    final_state = discrete_trajectory[i+tau] # Tracking the final state
    count_matrix[initial_state,final_state] += 1 # Inputting that transition into the transition matrix

print("Transition counts")
print(count_matrix)
print()

transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True) # Divide by total sum so row probs. add up to 1

print("Transition matrix")
print(transition_matrix)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(trajectory[:1000], color='blue', alpha=0.6)
ax1.set_title("Continuous Trajectory (First 1000 steps showing metastability)")
ax1.set_ylabel("Position (x)")

ax2.plot(discrete_trajectory[:1000], color='red', linestyle='--')
ax2.set_title("Discretized Trajectory (State 0 vs State 1)")
ax2.set_ylabel("Assigned State")
ax2.set_xlabel("Time Steps")
plt.tight_layout()

plt.savefig('markov_model.png',dpi=300,bbox_inches='tight')