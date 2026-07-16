import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from hmmlearn import hmm

""" 
Generate coupled trajectories (can be tweaked with gamma value)
"""



# STEP 1: Generate Coupled Trajectories

np.random.seed(42)
n_steps = 50000
dt = 0.01
noise_scaler = 0.4

# TUNABLE PARAMETER: 0.0 = Independent, 2.0 = Highly Correlated
gamma = int(input("gamma: "))

trajectory_X = np.zeros(n_steps)
trajectory_Y = np.zeros(n_steps)

# Start both in the left well
x = -1.0 
y = -1.0 

for t in range(1, n_steps):
    # Base double-well forces
    base_force_x = -4 * (x**3) + 4 * x
    base_force_y = -4 * (y**3) + 4 * y
    
    # Coupling force (Spring penalty)
    coupling_x = -gamma * (x - y)
    coupling_y = -gamma * (y - x)
    
    # Total forces
    force_x = base_force_x + coupling_x
    force_y = base_force_y + coupling_y
    
    # Update positions with independent noise
    x += force_x * dt + np.random.normal(0, noise_scaler)
    y += force_y * dt + np.random.normal(0, noise_scaler)
    
    trajectory_X[t] = x
    trajectory_Y[t] = y


# STEP 2: Discretize Both Spaces

kmeans_X = KMeans(n_clusters=2, random_state=42, n_init='auto')
discrete_X = kmeans_X.fit_predict(trajectory_X.reshape(-1, 1))

kmeans_Y = KMeans(n_clusters=2, random_state=42, n_init='auto')
discrete_Y = kmeans_Y.fit_predict(trajectory_Y.reshape(-1, 1))


# STEP 3: Create the Joint State Space

# Map (X, Y) pairs to a single state (0 to 3)
# (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
joint_discrete = discrete_X * 2 + discrete_Y





# # Plot a slice to visually verify correlation
# plt.figure(figsize=(10, 4))
# plt.plot(trajectory_X[:500], label="Process X", alpha=0.8)
# plt.plot(trajectory_Y[:500], label="Process Y", alpha=0.8)
# plt.title(f"Gamma = {gamma}")
# plt.legend()
# name = input("figname: ")
# plt.savefig(f"{name}",dpi=300,bbox_inches='tight')
# plt.close()

""" 
Testing different coupling methods
 """

# (Assume trajectory_X, trajectory_Y, discrete_X, discrete_Y, and joint_discrete 
# were generated here using the previous script with a specific gamma value)

tau = 50


# 1. Calculate Empirical Joint Matrix (The "True" Physics)

count_matrix_joint = np.zeros((4, 4))
for i in range(len(joint_discrete) - tau):
    count_matrix_joint[joint_discrete[i], joint_discrete[i + tau]] += 1

# Normalize rows to sum to 1 (add small epsilon to avoid divide-by-zero)
empirical_matrix = count_matrix_joint / (count_matrix_joint.sum(axis=1, keepdims=True) + 1e-9)


# 2. Calculate Kronecker Matrix (The "Independent" Assumption)

# Helper function to get 2x2 matrix from a 1D discrete trajectory
def get_transition_matrix(discrete_traj, lag, n_states=2):
    counts = np.zeros((n_states, n_states))
    for i in range(len(discrete_traj) - lag):
        counts[discrete_traj[i], discrete_traj[i + lag]] += 1
    return counts / (counts.sum(axis=1, keepdims=True) + 1e-9)

matrix_X = get_transition_matrix(discrete_X, lag=tau)
matrix_Y = get_transition_matrix(discrete_Y, lag=tau)

kronecker_matrix = np.kron(matrix_X, matrix_Y)


# 3. Calculate Divergence (Error)

# The Frobenius norm measures the total absolute difference between the matrices
error = np.linalg.norm(empirical_matrix - kronecker_matrix, ord='fro')
print(f"Coupling Strength (Gamma): {gamma}")
print(f"Matrix Divergence Error: {error:.4f}")


# # 4. Visualize with Heatmaps

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Plot Empirical
# sns.heatmap(empirical_matrix, annot=True, fmt=".2f", cmap="Blues", ax=axes[0], vmin=0, vmax=1, linewidths=1,linecolor='black',clip_on=False)
# axes[0].set_title(f"Empirical Matrix\n(True Physics at Gamma={gamma})")
# axes[0].set_xlabel("State at t + tau")
# axes[0].set_ylabel("State at t")

# # Plot Kronecker
# sns.heatmap(kronecker_matrix, annot=True, fmt=".2f", cmap="Reds", ax=axes[1], vmin=0, vmax=1, linewidths=1,linecolor='black',clip_on=False)
# axes[1].set_title("Kronecker Matrix\n(Assumed Independent)")
# axes[1].set_xlabel("State at t + tau")

# name = input("figname: ")
# plt.savefig(f"{name}.png",dpi=300,bbox_inches='tight')
# plt.close()


""" 
Hidden Markov Model testing
 """

# 1. Format the data for hmmlearn
# hmmlearn expects a 2D array where each row is a time step
# We use the joint_discrete array from your previous script (values 0, 1, 2, 3)
X_observed = joint_discrete.reshape(-1, 1)

# 2. Initialize the Hidden Markov Model
# We tell it to look for 2 hidden latent states.
# We use 'CategoricalHMM' because our emissions are discrete categories (0-3).
model = hmm.CategoricalHMM(n_components=2, random_state=42, n_iter=100)

# 3. Train the Model (The Reverse-Engineering Step)
# This runs the Baum-Welch (Expectation-Maximization) algorithm.
# It iteratively guesses the hidden transition matrix and emission matrix 
# until it finds the ones that most likely generated your observed data.
model.fit(X_observed)

# 4. Extract the Learned Physics
latent_transition_matrix = model.transmat_
emission_matrix = model.emissionprob_

print("--- The Hidden Latent Space Transitions ---")
print(latent_transition_matrix)
print("\n--- The Emission Probabilities ---")
print("(Rows = Hidden State, Columns = Joint State 0, 1, 2, 3)")
print(emission_matrix)

# 5. Decode the Hidden Trajectory (Viterbi Algorithm)
# Now that the model understands the rules, we ask it to look at your 
# observed sequence and guess exactly when the hidden state switched!
hidden_states_inferred = model.predict(X_observed)