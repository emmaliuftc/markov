import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

# ==========================================
# 1. Generate Latent-Coupled Data
# ==========================================
def generate_latent_data(rho, true_trans_mat, n_steps=10000):

    np.random.seed(42)
    
    # centers at (-1, -1)
    means = np.array([[-1.0, -1.0], 
                      [ 1.0,  1.0]])
    
    # Covariance matrix
    cov = np.array([[1.0, rho], 
                    [rho, 1.0]])
    
    hidden_states = np.zeros(n_steps, dtype=int)
    observed_data = np.zeros((n_steps, 2)) # Column 0: shape, Column 1: RNA
    
    # Generate the sequence
    current_state = 0
    for t in range(n_steps):
        # 1. Jump states based on the hidden transition matrix
        if np.random.rand() < true_trans_mat[current_state, 1 - current_state]:
            current_state = 1 - current_state
        hidden_states[t] = current_state
        
        # 2. Generate shape and RNA from the correlated gaussian
        observed_data[t] = np.random.multivariate_normal(means[current_state], cov)
        
    return hidden_states, observed_data

def generate_trajectory(n_steps=50000, dt=0.01, noise=0.4): 
    """Simulates the double-well particle."""
    x, y = -1.0, -1.0
    traj_X = np.zeros(n_steps)
    
    for t in range(1, n_steps):
        # Base double-well forces
        force_x = -4 * (x**3) + 4 * x
        
        # Update positions with independent noise
        x += force_x * dt + np.random.normal(0, noise)
        
        traj_X[t] = x
        
    return traj_X


# A. Generate the data
traj_X = generate_trajectory()

# B. Discretize (K-Means)
discrete_X = KMeans(n_clusters=2, random_state=42, n_init='auto').fit_predict(traj_X.reshape(-1, 1))

# Subsample the discretized data using a lag time (tau)
tau = 20

count_matrix = np.zeros((2, 2)) 
for i in range(len(discrete_X) - tau):
    count_matrix[discrete_X[i], discrete_X[i + tau]] += 1

# Normalize rows to sum to 1 (add small epsilon to avoid divide-by-zero)
empirical_matrix = count_matrix / (count_matrix.sum(axis=1, keepdims=True) + 1e-9)

print("Empirical matrix/actual dynamics of the system")
print(empirical_matrix)
# ^^ Generates the actual trans mat data

# true_rho = 0.60 
true_rho = float(input("rho: "))
hidden_states, X_observed = generate_latent_data(rho=true_rho,true_trans_mat=empirical_matrix)

# ==========================================
# 2.5 Visualize the Generated Data (Pre-HMM)
# ==========================================
print("Generating pre-HMM visualization...")

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot A: What the HMM sees (Unlabeled raw data)
# The HMM has to figure out how to separate this single blob into two states
axes1[0].scatter(X_observed[:, 0], X_observed[:, 1], alpha=0.15, s=10, color='gray')
axes1[0].set_title("What the HMM Sees\n(Unlabeled Continuous Data)")
axes1[0].set_xlabel("Shape Feature")
axes1[0].set_ylabel("RNA Feature")
axes1[0].grid(True, alpha=0.3)

# Plot B: What Nature knows (Ground Truth Colors)
# c=hidden_states colors the points by the true underlying biological state
axes1[1].scatter(X_observed[:, 0], X_observed[:, 1], c=hidden_states, 
                cmap='bwr', alpha=0.15, s=10)
axes1[1].set_title(f"Ground Truth\n(True Rho = {true_rho:.2f})")
axes1[1].set_xlabel("Shape Feature")
axes1[1].grid(True, alpha=0.3)

fig1.savefig(f"generated_data_{true_rho}.png",dpi=300,bbox_inches='tight')
# plt.close()



# Initialize the Continuous HMM
# covariance_type="full" tells the HMM to look for the off-diagonal correlation!
model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=42, n_iter=100)

# Fit directly on the raw continuous data (No K-means required!)
# X_joint = np.column_stack((discrete_X,X_observed))
# print(X_joint.shape)
print(X_observed.shape)
model.fit(X_observed)


# ==========================================
# 3. Did the HMM Reverse-Engineer the Biology?
# ==========================================
print(f"--- Ground Truth Correlation (Rho) ---")
print(f"Target: {true_rho:.2f}")

print("\n--- HMM Learned Covariance Matrices ---")
# Extract the learned covariances for State 0 and State 1
learned_covariances = model.covars_

for i in range(2):
    print(f"\nHidden State {i}:")
    print(np.round(learned_covariances[i], 3))

print("Model transmat")
print(model.transmat_)


# ==========================================
# 4. Decode the Hidden States (Viterbi Algorithm)
# ==========================================
# Ask the model to guess which state each point belongs to
predicted_states = model.predict(X_observed)

# Fix Label Switching (just in case the HMM named them backward)
# We know State 1 should have a higher mean than State 0
if model.means_[0, 0] > model.means_[1, 0]:
    predicted_states = 1 - predicted_states

# ==========================================
# 5. Visualize the Probability Clouds
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Plot 1: Ground Truth (Nature)
# c=hidden_states colors the points by the actual generative rules
axes[0].scatter(X_observed[:, 0], X_observed[:, 1], c=hidden_states, 
                cmap='bwr', alpha=0.15, s=10)
axes[0].set_title("Ground Truth\n(True Hidden States)")
axes[0].set_xlabel("Shape Feature")
axes[0].set_ylabel("RNA Feature")
axes[0].grid(True, alpha=0.3)

# Plot 2: HMM Inference (The Detective)
# c=predicted_states colors the points by what the HMM figured out
axes[1].scatter(X_observed[:, 0], X_observed[:, 1], c=predicted_states, 
                cmap='bwr', alpha=0.15, s=10)
axes[1].set_title(f"HMM Inference\n(Learned Rho ≈ {learned_covariances[0][0,1]:.2f})")
axes[1].set_xlabel("Shape Feature")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# name = input("figname: ")
# plt.savefig(f"{name}.png",dpi=300,bbox_inches='tight')
plt.savefig(f"covariance_{true_rho}.png",dpi=300,bbox_inches='tight')
plt.close()


mse = np.mean((hidden_states - predicted_states)**2)
print(f"Mean squared error between hidden and predicted: {mse}")
print(f"For {hidden_states.shape}")



# ---------------------------------------------------------
# Extract the learned rulebook for Hidden State 0
# ---------------------------------------------------------
state = 0
mean_vector = model.means_[state]
cov_matrix = model.covars_[state]

# =========================================================
# Goal A: Generate ACTUAL VALUES (Sampling)
# =========================================================
# Draw 5 random simulated cells belonging to State 0
print(f"--- Generating 5 Simulated Cells from State {state} ---")
for i in range(5):
    # This acts as the "emission" step
    simulated_emission = np.random.multivariate_normal(mean_vector, cov_matrix)
    print(f"Cell {i+1} [Shape, RNA]: {np.round(simulated_emission, 3)}")

# =========================================================
# Goal B: Calculate EMISSION PROBABILITIES (Density)
# =========================================================
test_point = np.array([-2.0, -2.0])

# Calculate how perfectly this point fits into cloud
prob_density = multivariate_normal.pdf(test_point, mean=mean_vector, cov=cov_matrix)

print(f"\n--- Emission Probability Density ---")
print(f"How strongly does State {state} emit the coordinate {test_point}?")
print(f"Density: {prob_density:.4f}")



# =========================================================
# 6. Visualize the Probability Density Contours
# =========================================================
print("Generating Probability Density Contours...")

# 1. Define the boundaries of your "map" (e.g., from -4 to 4)
x_min, x_max = -4.0, 4.0
y_min, y_max = -4.0, 4.0

# 2. Create the invisible Meshgrid (resolution of 0.05)
X_grid, Y_grid = np.mgrid[x_min:x_max:0.05, y_min:y_max:0.05]

# Stack the X and Y grids into a 3D array of [Shape, RNA] coordinates
# This creates a grid of points ready to be evaluated by the PDF
positions = np.dstack((X_grid, Y_grid))

# 3. Create the mathematical PDF objects using the HMM's learned rules
rv_state_0 = multivariate_normal(model.means_[0], model.covars_[0])
rv_state_1 = multivariate_normal(model.means_[1], model.covars_[1])

# 4. Calculate the exact density for every single point on the grid
density_state_0 = rv_state_0.pdf(positions)
density_state_1 = rv_state_1.pdf(positions)

# 5. Plot the Contours
fig, ax = plt.subplots(figsize=(7, 6))

# Plot State 0 (Blue rings)
# levels=5 draws 5 distinct topographical rings
ax.contour(X_grid, Y_grid, density_state_0, levels=5, colors='blue', alpha=0.7, linewidths=2)

# Plot State 1 (Red rings)
ax.contour(X_grid, Y_grid, density_state_1, levels=5, colors='red', alpha=0.7, linewidths=2)

ax.set_title("HMM Learned Probability Densities")
ax.set_xlabel("Shape Feature")
ax.set_ylabel("RNA Feature")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"density_wells_{true_rho}.png",dpi=300,bbox_inches="tight")


###################

# Assuming 'model' is your already-trained GaussianHMM

# Ask the model to dream up 1000 brand new time steps
# It returns BOTH the observable continuous data and the hidden state timeline!
X_synthetic, hidden_synthetic = model.sample(n_samples=1000)

# X_synthetic is a (1000, 2) array of your fake [Shape, RNA] values
synthetic_shape = X_synthetic[:, 0]
synthetic_rna   = X_synthetic[:, 1]

print("Successfully generated synthetic trajectory!")




#########



# =========================================================
# 7. Visualize the Synthetic Trajectories over Time
# =========================================================
print("Generating Timeline Plots...")

# Create a figure with 2 vertically stacked subplots that share the same X-axis (Time)
# We will just plot the first 200 steps so the jumps are clearly visible, 
# rather than zooming out so far that it looks like static.
n_plot_steps = 200
time_axis = range(n_plot_steps)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# ---------------------------------------------------------
# Plot A: The Hidden State Timeline (The "Master Clock")
# ---------------------------------------------------------
# We use a 'step' plot because the biological state is discrete (it instantly snaps from 0 to 1)
axes[0].step(time_axis, hidden_synthetic[:n_plot_steps], where='post', color='black', linewidth=2)
axes[0].set_title("The Hidden Latent Space")
axes[0].set_ylabel("Hidden State")
axes[0].set_yticks([0, 1])
axes[0].grid(True, alpha=0.3)

# ---------------------------------------------------------
# Plot B: The Continuous Emissions (Shape and RNA)
# ---------------------------------------------------------
axes[1].plot(time_axis, X_synthetic[:n_plot_steps, 0], label="Shape Feature", color='blue', alpha=0.8, linewidth=1.5)
axes[1].plot(time_axis, X_synthetic[:n_plot_steps, 1], label="RNA Feature", color='red', alpha=0.8, linewidth=1.5)
axes[1].set_title("Observable Data (Emissions)")
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Continuous Value")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"full_plot_{true_rho}.png",dpi=300,bbox_inches="tight")