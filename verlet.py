import numpy as np
from hmmlearn import hmm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import math

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

def double_well_force(r):
    """Deterministic force"""
    # print(r)
    return (-4 * r**3) + (4 * r)

def simulate_gjf_trajectory(m, alpha, T, dt, steps, r0, v0):
    """
    Generates a trajectory using the GJ-F modified Verlet algorithm.
    """
    # 1. Initialize the Mersenne Twister Random Number Generator
    mt_rng = np.random.Generator(np.random.MT19937(seed=42))
    
    # 2. Boltzmann constant 
    kB = 1.0
    
    # 3. Pre-calculate GJ-F damping coefficients
    b = 1.0 / (1.0 + (alpha * dt) / (2.0 * m))
    a = (1.0 - ((alpha * dt)/(2.0 * m)) / (1.0 + ((alpha * dt)/(2.0 * m))))
    
    # 4. Noise standard deviation based on Fluctuation-Dissipation theorem
    variance = 2.0 * alpha * kB * T * dt
    std_dev = np.sqrt(variance)
    
    # 5. Initialize trajectory arrays
    r_traj = np.zeros(steps)
    v_traj = np.zeros(steps)
    t_traj = np.zeros(steps)
    
    r = r0
    r_prev = r0 - v0 * dt
    v = v0
    f = double_well_force(r)
    
    beta_curr = mt_rng.normal(loc=0.0, scale=std_dev)

    # 6. Integration loop
    for n in range(steps):
        # Store current state
        r_traj[n] = r
        t_traj[n] = n * dt
        
        # Generate random noise beta^n (Mersenne Twister)
        beta_next = mt_rng.normal(loc=0.0, scale=std_dev)

        # Calculate new force at the updated position
        f_curr = double_well_force(r) * 10

        # Update Position
        term1 = 2 * b * r
        term2 = a * r_prev
        term3 = b * (dt**2) * f_curr / m
        term4 = (b * dt / (2.0 * m)) * (beta_curr + beta_next)
        
        # r_new = term1 - term2 + term3 + term4
        r_new = term1 - term2 - term3 + term4
        

        # Advance variables for next step
        r_prev = r
        r = r_new
        beta_curr = beta_next
        
    return t_traj, r_traj

def generate_trajectory(n_steps, dt, alpha, m, T, init_pos): 
    """Simulates the double-well particle."""
    x = init_pos
    traj_X = np.zeros(n_steps)
    
    # 1. Initialize the Mersenne Twister Random Number Generator
    mt_rng = np.random.Generator(np.random.MT19937(seed=42))
    
    # 2. Boltzmann constant 
    kB = 1.0
    
    # 3. Pre-calculate GJ-F damping coefficients
    b = 1.0 / (1.0 + (alpha * dt) / (2.0 * m))
    
    # 4. Noise standard deviation based on Fluctuation-Dissipation theorem
    variance = 2.0 * alpha * kB * T * dt
    std_dev = np.sqrt(variance)
    
    
    beta_curr = mt_rng.normal(loc=0.0, scale=1)

    for t in range(n_steps):
        # Base double-well forces
        force_x = double_well_force(x)

        beta_next = mt_rng.normal(loc=0.0, scale=1)

        # Update positions with independent noise
        noise_term = std_dev * beta_next * dt # (b * dt / (2.0 * m)) * (beta_curr + beta_next)
        # noise_term = (b * dt / (2.0 * m)) * (beta_curr + beta_next)
        

        x += force_x * alpha * dt + noise_term
        
        beta_curr = beta_next

        traj_X[t] = x
        
    return traj_X

# ==========================================
# Run the Simulation
# ==========================================
mass = 1.0
drag_coeff = 1
print(f"drag coeff : {drag_coeff}",flush=True)
temperature = 40000
time_step = 0.01
num_steps = 500000

# Initial conditions (start near one of the wells: r = -1 or r = 1)
initial_position = 0
initial_velocity = 0

# time, positions = simulate_gjf_trajectory(
#     m=mass, 
#     alpha=drag_coeff, 
#     T=temperature, 
#     dt=time_step, 
#     steps=num_steps, 
#     r0=initial_position, 
#     v0=initial_velocity
# )

euler_positions = generate_trajectory(n_steps=num_steps,dt=time_step, alpha = drag_coeff,m=mass, T=temperature, init_pos=initial_position)

print("Plotting the Trajectory")

plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True)

print("Plotting the Trajectory")
# plt.plot(time, positions, label="Position (r)", color="blue")
# plt.title(f"GJ-F Trajectory in a Double-Well Potential (alpha={drag_coeff})")
plt.xlabel("Time")
plt.ylabel("Position $r$")
plt.axhline(1, color='gray', linestyle='--', label="Right Well")
plt.axhline(-1, color='gray', linestyle='--', label="Left Well")
plt.legend(loc="upper right")
# plt.savefig(f"verlet_alpha{drag_coeff}.png",dpi=300,bbox_inches="tight")


# plt.plot(euler_positions[0:200], label="Position (r)", color="blue")
plt.title(f"Euler Trajectory in a Double-Well Potential (alpha={drag_coeff},t={temperature},timestep={time_step})")

plt.savefig(f"v-euler_alpha{drag_coeff}.png",dpi=300,bbox_inches="tight")
# plt.close()


""" 



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plot 1
ax1.plot(time, positions, label=f"G-JF {drag_coeff}", color="blue")
ax1.set_title(f"GJ-F Trajectory 1 (Seed: 42, Drag: {drag_coeff})")
ax1.set_ylabel("Position $r$")
ax1.axhline(1, color='gray', linestyle='--', alpha=0.7, label="Right Well")
ax1.axhline(-1, color='gray', linestyle='--', alpha=0.7, label="Left Well")
ax1.legend(loc="upper right")


# Plot 2
ax2.plot(time, euler_positions, label="Euler", color="crimson")
ax2.set_title(f"Euler Trajectory (rho=0.4)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Position $r$")
ax2.axhline(1, color='gray', linestyle='--', alpha=0.7, label="Right Well")
ax2.axhline(-1, color='gray', linestyle='--', alpha=0.7, label="Left Well")
ax2.legend(loc="upper right")


# Adjust layout so titles and labels don't overlap
plt.tight_layout()
# plt.show()

plt.savefig(f"trajectory_comparison{drag_coeff}.png",dpi=300,bbox_inches="tight")

 """

# B. Discretize (K-Means)
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
discrete_X = kmeans.fit_predict(euler_positions.reshape(-1, 1))

print("Plotting the Discretized trajectory")
# plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True)
# plt.xlabel("Time")
# plt.ylabel("Position $r$")

plt.plot(euler_positions[0:200], label="Position (r)", color="gray", alpha=0.5)

graphing_X = 1 - 2 * discrete_X 

plt.step(range(len(graphing_X))[0:200], graphing_X[0:200], label="Position (r)", color="blue",linestyle="--")
plt.title(f"Discretized Trajectory (alpha={drag_coeff},t={temperature},timestep={time_step})")

plt.savefig(f"v-euler_kmeans_alpha{drag_coeff}.png",dpi=300,bbox_inches="tight")
plt.close()

print(kmeans.cluster_centers_)

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



# MICROSTATE SECTION

kmeans_micro = KMeans(n_clusters=16, random_state=42, n_init='auto')
discrete_X_micro = kmeans_micro.fit_predict(euler_positions.reshape(-1, 1))

# Subsample the discretized data using a lag time (tau)
tau = 20

count_matrix_micro = np.zeros((16,16)) 
for i in range(len(discrete_X_micro) - tau):
    count_matrix_micro[discrete_X_micro[i], discrete_X_micro[i + tau]] += 1

# Normalize rows to sum to 1 (add small epsilon to avoid divide-by-zero)
empirical_matrix_micro = count_matrix_micro / (count_matrix_micro.sum(axis=1, keepdims=True) + 1e-9)

print("MICRO: Empirical matrix/actual dynamics of the system")
print(empirical_matrix_micro)
# ^^ Generates the actual trans mat data






import scipy.linalg as la
import matplotlib.pyplot as plt

print("\n==========================================")
print(" Markov Chain Analysis & Plotting")
print("==========================================")

# ---------------------------------------------------------
# A. Spectral Gap & Implied Timescales
# ---------------------------------------------------------
evals, evecs = la.eig(empirical_matrix_micro)
evals_sorted = np.sort(np.abs(evals))[::-1]

# The first eigenvalue (index 0) is lambda_1 = 1.
# Process indices start at 2 (which corresponds to lambda_2, lambda_3, etc.)
process_indices = []
implied_timescales = []

for i in range(1, len(evals_sorted)):
    lam = evals_sorted[i]
    if 0 < lam < 1:
        t_i = -tau / np.log(lam)
        implied_timescales.append(t_i)
        process_indices.append(i + 1) # Start index at 2
    else:
        # If eigenvalue is 0 or negative, implied timescale is undefined/zero
        implied_timescales.append(0)
        process_indices.append(i + 1)

print("--- Spectral Analysis ---")
print(f"Eigenvalues: {np.round(evals_sorted, 4)}")
if len(implied_timescales) > 0:
    print(f"Longest Implied Timescale (t_2): {implied_timescales[0]:.2f}")

# --- Plot 1: Implied Timescales (Screenshot Style) ---
fig, ax = plt.subplots(figsize=(8, 4))

# Maroon line with circular markers
ax.plot(process_indices, implied_timescales, marker='o', linestyle='-', 
        color='#800000', markersize=6, linewidth=2)

ax.set_title(f"Spectral Gap Analysis (Lag Time $\\tau = {tau}$)")
ax.set_xlabel("Timescale ($i$)")
ax.set_ylabel("Implied timescale $t_i$ (steps)")

# Format the grid to match the screenshot (light dashed lines)
ax.grid(True, which='both', color='lightgray', linestyle='--', alpha=0.7)

# Ensure x-axis ticks are integers
if len(process_indices) > 0:
    ax.set_xticks(range(min(process_indices), max(process_indices) + 1))

plt.savefig(f"v-euler_implied_timescales_tau{tau}.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: v-euler_implied_timescales_tau{tau}.png")


# ---------------------------------------------------------
# B. Chapman-Kolmogorov Test
# ---------------------------------------------------------
print("\n--- Chapman-Kolmogorov Test ---")

def estimate_trans_mat(data, lag_steps):
    c_mat = np.zeros((2, 2)) 
    for i in range(len(data) - lag_steps):
        c_mat[data[i], data[i + lag_steps]] += 1
    return c_mat / (c_mat.sum(axis=1, keepdims=True) + 1e-9)

k_multipliers = [1, 2, 3, 5, 8, 12]

# Arrays to store data for plotting
ck_errors = []
p00_direct = []
p00_predicted = []
p11_direct = []
p11_predicted = []

for k in k_multipliers:
    P_direct = estimate_trans_mat(discrete_X, lag_steps=tau * k)
    P_predicted = np.linalg.matrix_power(empirical_matrix, k)
    
    # Calculate Frobenius Error
    error = la.norm(P_direct - P_predicted, ord='fro')
    ck_errors.append(error)
    
    # Store self-transition probabilities for states 0 and 1
    p00_direct.append(P_direct[0, 0])
    p00_predicted.append(P_predicted[0, 0])
    p11_direct.append(P_direct[1, 1])
    p11_predicted.append(P_predicted[1, 1])
    
    print(f"\nLag multiple k={k} (Total steps = {tau * k})")
    print(f"Frobenius Norm Error: {error:.5f}")

# --- Plot 2: Chapman-Kolmogorov Probabilities ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot A: Self-Transitions P(0,0) and P(1,1)
ax1.plot(k_multipliers, p00_direct, 'bo-', label="$P(0 \u2192 0)$ Direct")
ax1.plot(k_multipliers, p00_predicted, 'b--', label="$P(0 \u2192 0)$ Predicted", alpha=0.7)
ax1.plot(k_multipliers, p11_direct, 'ro-', label="$P(1 \u2192 1)$ Direct")
ax1.plot(k_multipliers, p11_predicted, 'r--', label="$P(1 \u2192 1)$ Predicted", alpha=0.7)
ax1.set_title("C-K Test: Self-Transition Probabilities")
ax1.set_xlabel("Lag Multiplier $k$")
ax1.set_ylabel("Probability")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot B: Error over time
ax2.plot(k_multipliers, ck_errors, 'ko-', linewidth=2)
ax2.set_title("C-K Test: Frobenius Matrix Error")
ax2.set_xlabel("Lag Multiplier $k$")
ax2.set_ylabel("Error Magnitude")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"v-euler_chapman_kolmogorov_test_tau{tau}.png", dpi=300, bbox_inches="tight")
plt.close(fig) # Clear memory
print(f"Saved: v-euler_chapman_kolmogorov_test_tau{tau}.png")












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

fig1.savefig(f"v-euler_generated_data_{true_rho}.png",dpi=300,bbox_inches='tight')
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
plt.savefig(f"v-euler_covariance_{true_rho}.png",dpi=300,bbox_inches='tight')
plt.close()


mse = np.mean((hidden_states - predicted_states)**2)
print(f"Mean squared error between hidden and predicted: {mse}")
print(f"For {hidden_states.shape}")



# ---------------------------------------------------------
# Extract the learned rulebook for Hidden State 0
# ---------------------------------------------------------
state = 1
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
test_point = np.array([3.0, 3.0])

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
plt.savefig(f"v-euler_density_wells_{true_rho}.png",dpi=300,bbox_inches="tight")


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

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

# ---------------------------------------------------------
# Plot A: The Hidden State Timeline (The "Master Clock")
# ---------------------------------------------------------
# We use a 'step' plot because the biological state is discrete (it instantly snaps from 0 to 1)
axes[0].step(time_axis, hidden_synthetic[:n_plot_steps], where='post', color='black', linewidth=2)
axes[0].set_title("Hidden Latent Space")
axes[0].set_ylabel("Hidden State")
axes[0].set_yticks([0, 1])
axes[0].grid(True, alpha=0.3)

# ---------------------------------------------------------
# Plot B: The Continuous Emissions (Shape and RNA)
# ---------------------------------------------------------
axes[1].plot(time_axis, X_synthetic[:n_plot_steps, 0], label="Shape", color='blue', alpha=0.8, linewidth=1.5)
axes[1].plot(time_axis, X_synthetic[:n_plot_steps, 1], label="RNA", color='red', alpha=0.8, linewidth=1.5)
axes[1].set_title("Synthetic Emissions")
# axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Continuous Value")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_axis, X_observed[:n_plot_steps, 0], label="Shape", color='blue', alpha=0.8, linewidth=1.5)
axes[2].plot(time_axis, X_observed[:n_plot_steps, 1], label="RNA", color='red', alpha=0.8, linewidth=1.5)
axes[2].set_title("Real Data")
axes[2].set_xlabel("Time Step")
axes[2].set_ylabel("Continuous Value")
axes[2].legend(loc="upper right")

plt.savefig(f"v-euler_full_plot_{true_rho}.png",dpi=300,bbox_inches="tight")


ideal_emission = model.means_
print(f"ideal emission: {ideal_emission[0]}, {ideal_emission[1]}")




















# =========================================================
# 8. Cross-Modality Inference (Imputing RNA from Shape)
# =========================================================
print("\nRunning Cross-Modality Inference...")

# 1. Hide the RNA Data (Simulate a Shape-only experiment)
# We will just test this on the first 300 steps for clear visualization
n_test_steps = 300
shape_only_input = X_observed[:n_test_steps, 0].reshape(-1, 1) # Only Column 0
true_rna_hidden = X_observed[:n_test_steps, 1]                 # The actual RNA (for the answer key)

# 2. Create the 1D Detective HMM
shape_only_hmm = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="")

# Copy the exact Hidden Space rules (The Clock)
shape_only_hmm.transmat_ = model.transmat_
shape_only_hmm.startprob_ = model.startprob_

# Copy ONLY the Shape Means (Column 0)
shape_only_hmm.means_ = model.means_[:, 0].reshape(-1, 1)

# Copy ONLY the Shape Variances (Row 0, Col 0 of the Covariance Matrices)
shape_var_0 = model.covars_[0][0, 0]
shape_var_1 = model.covars_[1][0, 0]
shape_only_hmm.covars_ = np.array([[shape_var_0], [shape_var_1]])

# 3. Decode the Timeline using ONLY Shape
imputed_timeline = shape_only_hmm.predict(shape_only_input)

# Fix Label Switching for the 1D model just in case
if shape_only_hmm.means_[0, 0] > shape_only_hmm.means_[1, 0]:
    imputed_timeline = 1 - imputed_timeline

# =========================================================
# 4. Hallucinate the RNA Data (Rigorous Conditional Sampling)
# =========================================================
imputed_rna_values = np.zeros(n_test_steps)

np.random.seed(42) # Keep noise consistent for comparison

for t, state in enumerate(imputed_timeline):
    # 1. Extract the Master Rules for this specific state
    mu_shape = model.means_[state, 0]
    mu_rna   = model.means_[state, 1]
    
    var_shape = model.covars_[state][0, 0]
    var_rna   = model.covars_[state][1, 1]
    cov_xy    = model.covars_[state][0, 1] # The off-diagonal correlation
    
    # 2. Get the EXACT Shape value we observed at this millisecond
    # (shape_only_input is a 2D array, so we pull the scalar value)
    observed_shape = shape_only_input[t, 0] 
    
    # 3. Calculate the Conditional Mean (The Shift)
    cond_mean_rna = mu_rna + (cov_xy / var_shape) * (observed_shape - mu_shape)
    
    # 4. Calculate the Conditional Variance (The Squeeze)
    cond_var_rna = var_rna - ((cov_xy**2) / var_shape)
    
    # Ensure variance doesn't hit absolute zero due to floating point math
    cond_var_rna = max(cond_var_rna, 1e-9) 
    cond_std_rna = np.sqrt(cond_var_rna)
    
    # 5. Draw the random value from the highly precise conditional curve
    imputed_rna_values[t] = np.random.normal(loc=cond_mean_rna, scale=cond_std_rna)

print("Successfully imputed RNA using precise Conditional Normal Distributions!")
    

# 5. Visualize the Imputation Accuracy
fig, ax = plt.subplots(figsize=(12, 4))

# subset = len(n_test_steps)
subset = 200

# Plot the actual, noisy RNA that the model was NOT allowed to see
ax.plot(range(n_test_steps)[0:subset], true_rna_hidden[0:subset], label="RNA (Hidden from Model)", color="gray", alpha=0.5, linewidth=1)

# Plot the clean, imputed RNA that the model guessed based purely on Shape
ax.plot(range(n_test_steps)[0:subset], imputed_rna_values[0:subset], label="RNA (Guessed from Shape)", color="red", linewidth=1.25)

ax.set_title(f"Cross Inference (Rho = {true_rho:.2f})")
ax.set_xlabel("Time Step")
ax.set_ylabel("RNA Feature Level")
ax.legend(loc="upper right")
# ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"v-euler_imputation_{true_rho}.png", dpi=300, bbox_inches="tight")
print("Imputation complete. Saved as v-euler_imputation plot.")

print(f"mse: {np.mean((true_rna_hidden[0:subset] - imputed_rna_values[0:subset])**2)}")
print(f"rmse: {math.sqrt(np.mean((true_rna_hidden[0:subset] - imputed_rna_values[0:subset])**2))}")