import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate a particle in a double-well potential; basically goes towards -1 or 1

np.random.seed(42)
n_steps = 500000
dt = 0.01 
noise_scalar = 0.25

trajectory = np.zeros(n_steps)
x = -1.0 # Start in the left well

for t in range(1, n_steps):
    force = -4 * (x**3) + 4 * x
    # Update poistion -- deterministic force + some random noise
    x += force * dt + np.random.normal(0,noise_scalar)
    trajectory[t] = x

X = trajectory.reshape(-1, 1)

# Compute transition matrix

def get_transition_matrix(traj, tau, n_states):
    count_matrix = np.zeros((n_states, n_states))
    for i in range(len(traj) - tau):
        initial_state = traj[i] 
        final_state = traj[i+tau] 
        count_matrix[initial_state, final_state] += 1 

    row_sums = count_matrix.sum(axis=1, keepdims=True)
    # Normalize matrix
    # Use np.maximum to avoid dividing by zero if a state is empty
    transition_matrix = count_matrix / np.maximum(row_sums, 1e-10) 
    return transition_matrix





# ==========================================
# STEP 2: Over-cluster into Microstates
# ==========================================
print("Clustering into 100 microstates...")
n_microstates = 100
kmeans = KMeans(n_clusters=n_microstates, random_state=42, n_init='auto')
micro_trajectory = kmeans.fit_predict(X)

# ==========================================
# STEP 3: The Implied Timescale Sweep (Validation)
# ==========================================
print("Running implied timescale sweep...")
lag_times = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
implied_timescales = []
valid_lags = []

for tau in lag_times:
    T_tau = get_transition_matrix(micro_trajectory, tau, n_microstates)
    evals = np.linalg.eigvals(T_tau)
    evals = np.sort(np.real(evals))[::-1]
    
    if len(evals) > 1 and 0 < evals[1] < 1:
        lambda_2 = evals[1]
        t2 = -tau / np.log(lambda_2)
        implied_timescales.append(t2)
        valid_lags.append(tau)

# Plot 1: Implied Timescales
plt.figure(figsize=(8, 5))
plt.plot(valid_lags, implied_timescales, marker='o', linewidth=2, color='navy')
plt.fill_between(valid_lags, 0, valid_lags, color='gray', alpha=0.2, label='Blind Spot (t < tau)')
plt.title("Implied Timescale vs. Lag Time (100 Microstates)")
plt.xlabel(r"Lag Time $\tau$ (steps)")
plt.ylabel(r"Implied Timescale $t_2$ (steps)")
plt.xscale('log')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("pipeline_1_timescales.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'pipeline_1_timescales.png'")

# ==========================================
# STEP 4: The Anchor (Choose the perfect Lag Time)
# ==========================================
# Based on looking at the plateau, tau=50 is a great choice.
chosen_tau = 50  
print(f"Building definitive matrix at lag time tau = {chosen_tau}...")
T_definitive = get_transition_matrix(micro_trajectory, chosen_tau, n_microstates)

# ==========================================
# NEW STEP: Find 'N' using the Spectral Gap
# ==========================================
print("Analyzing the Spectral Gap to find the number of macrostates...")

# 1. Get eigenvalues of the definitive matrix
eigenvalues, _ = np.linalg.eig(T_definitive)
evals = np.sort(np.real(eigenvalues))[::-1]

# 2. Convert the top 15 eigenvalues into implied timescales
# (Skipping the first eigenvalue because lambda_1 = 1, which means t_1 = infinity)
timescales = []
for val in evals[1:16]: 
    if 0 < val < 1:
        timescales.append(-chosen_tau / np.log(val))
    else:
        timescales.append(0) # Safety catch for bad values

# 3. Plot them to look for the "cliff"
plt.figure(figsize=(8, 4))
plt.plot(range(2, len(timescales) + 2), timescales, marker='o', color='darkred', linewidth=2)
plt.title(f"Spectral Gap Analysis (Lag Time $\\tau = {chosen_tau}$)")
plt.xlabel(r"Process Index ($i$)")
plt.ylabel(r"Implied Timescale $t_i$ (steps)")
plt.xticks(range(2, 17))
plt.grid(True, alpha=0.4, linestyle='--')

plt.tight_layout()
plt.savefig("spectral_gap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'spectral_gap.png' - Look for the massive drop!")

# ==========================================
# STEP 5: Schütte's Lumping (Translation to Biology)
# ==========================================
print("Performing PCCA (Micro to Macro) mapping...")
eigenvalues, eigenvectors = np.linalg.eig(T_definitive)

# Sort descending to grab the 2nd eigenvector
sort_indices = np.argsort(np.real(eigenvalues))[::-1]
eigenvectors = np.real(eigenvectors[:, sort_indices])
psi_2 = eigenvectors[:, 1]

# Map microstates to 2 macrostates based on eigenvector signs
macrostate_mapping = (psi_2 > 0).astype(int)
macro_trajectory = np.array([macrostate_mapping[state] for state in micro_trajectory])

# # --- ALIGNMENT FIX ---
# # Check where Macrostate 1 physically sits in the continuous data
# # If Macrostate 1 is mostly on the negative side (x < 0), flip the 0s and 1s
# if np.mean(X[macro_trajectory == 1]) < 0:
#     macro_trajectory = 1 - macro_trajectory
    
# Plot 2: PCCA Mapping
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graph A: Eigenvector vs Position
cluster_centers = kmeans.cluster_centers_.flatten()
ax1.scatter(cluster_centers, psi_2, c=macrostate_mapping, cmap='bwr', edgecolor='k', s=50)
ax1.axhline(0, color='black', linestyle='--')
ax1.set_title("Schütte's Mapping: Eigenvector vs. Position")
ax1.set_xlabel("Microstate Physical Position (x)")
ax1.set_ylabel(r"Eigenvector Value ($\psi_2$)")
ax1.grid(True, alpha=0.3)

# Graph B: The Final Macro Trajectory
ax2.plot(trajectory[:5000], color='gray', alpha=0.5, label='Continuous Position (x)')
# Scale macrostate 0/1 to -1/1 for visual overlay
ax2.plot(macro_trajectory[:5000] * 2 - 1, color='black', linestyle='--', label='Macrostate (Dynamical)')
ax2.set_title("The Final 2-State Macro Trajectory")
ax2.set_xlabel("Time (steps)")
ax2.set_ylabel("Position / State")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pipeline_2_macrostates.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 'pipeline_2_macrostates.png'")
print("Pipeline complete!")












""" 
# Find states

n_states = 2
kmeans = KMeans(n_clusters=n_states, random_state=42, n_init="auto")
discrete_trajectory = kmeans.fit_predict(X)


# Compute transition matrix

def get_transition_matrix(traj, tau, n_states):
    count_matrix = np.zeros((n_states,n_states))
    for i in range(len(traj) - tau):
        initial_state = traj[i] # Tracking the intial state
        final_state = traj[i+tau] # Tracking the final state
        count_matrix[initial_state,final_state] += 1 # Inputting that transition into the transition matrix

    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True) 
    # Divide by total sum so row probs. add up to 1
    return transition_matrix


# Calculate timescales at different lag times

lag_times = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
implied_timescales = []
valid_lags = []

for tau in lag_times:
    T_tau = get_transition_matrix(discrete_trajectory, tau, n_states)
    evals = np.linalg.eigvals(T_tau)
    evals = np.sort(np.real(evals))[::-1]

    if len(evals) > 1 and 0 < evals[1] < 1:
        lambda_2 = evals[1]
        t2 = -tau / np.log(lambda_2)
        implied_timescales.append(t2)
        valid_lags.append(tau)

        

if not valid_lags:
    print("Warning: No valid implied timescales found! The particle likely got stuck in a single state.")
else:


    # --- Plot 1: Implied Timescales ---
    plt.figure(figsize=(8, 5))
    plt.plot(valid_lags, implied_timescales, marker='o', linewidth=2, color='navy')
    plt.fill_between(valid_lags, 0, valid_lags, color='gray', alpha=0.2, label='Blind Spot (t < tau)')
    plt.title("Implied Timescale vs. Lag Time")
    plt.xlabel(r"Lag Time $\tau$ (steps)")
    plt.ylabel(r"Implied Timescale $t_2$ (steps)")
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save as PNG instead of plt.show()
    plt.savefig("implied_timescales.png", dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free up memory
    print("Saved 'implied_timescales.png'") """






# # ==========================================
# # STEP 3: Chapman-Kolmogorov (CK) Test
# # ==========================================
# # We choose a base lag time from the "flat" region of the timescale plot
# base_tau = 10
# k_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# # 1. The Markov Model prediction: [T(tau)]^k
# T_base = get_transition_matrix(discrete_trajectory, base_tau, n_states)

# predicted_T00 = []
# predicted_T11 = []
# empirical_T00 = []
# empirical_T11 = []

# for k in k_steps:
#     # Model Prediction: Matrix power
#     T_pred = np.linalg.matrix_power(T_base, k)
#     predicted_T00.append(T_pred[0, 0])
#     predicted_T11.append(T_pred[1, 1])
    
#     # Empirical Data: Actual transitions at lag time (k * base_tau)
#     T_emp = get_transition_matrix(discrete_trajectory, k * base_tau, n_states)
#     empirical_T00.append(T_emp[0, 0])
#     empirical_T11.append(T_emp[1, 1])

# # --- Plot 2: CK Test ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# x_axis = [k * base_tau for k in k_steps]

# # Plot State 0 -> State 0
# ax1.plot(x_axis, predicted_T00, 'k--', label='Model Prediction $[T(\\tau)]^k$')
# ax1.plot(x_axis, empirical_T00, 'ro', label='Empirical Data $T(k\\tau)$')
# ax1.set_title("Self-Transition: State 0 to State 0")
# ax1.set_xlabel("Time (steps)")
# ax1.set_ylabel("Probability")
# ax1.legend()
# ax1.grid(True, alpha=0.5)

# # Plot State 1 -> State 1
# ax2.plot(x_axis, predicted_T11, 'k--', label='Model Prediction $[T(\\tau)]^k$')
# ax2.plot(x_axis, empirical_T11, 'bo', label='Empirical Data $T(k\\tau)$')
# ax2.set_title("Self-Transition: State 1 to State 1")
# ax2.set_xlabel("Time (steps)")
# ax2.set_ylabel("Probability")
# ax2.legend()
# ax2.grid(True, alpha=0.5)

# plt.suptitle(f"Chapman-Kolmogorov Test (Base Lag Time $\\tau = {base_tau}$)", fontsize=14)
# plt.tight_layout()

# # Save as PNG instead of plt.show()
# plt.savefig("ck_test.png", dpi=300, bbox_inches='tight')
# plt.close()
# print("Saved 'ck_test.png'")















# tau = 10 # the transition/lag time in the matrix
# count_matrix = np.zeros((n_states,n_states))

# for i in range(len(discrete_trajectory) - tau):
#     initial_state = discrete_trajectory[i] # Tracking the intial state
#     final_state = discrete_trajectory[i+tau] # Tracking the final state
#     count_matrix[initial_state,final_state] += 1 # Inputting that transition into the transition matrix

# print("Transition counts")
# print(count_matrix)
# print()

# transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True) # Divide by total sum so row probs. add up to 1

# print("Transition matrix")
# print(transition_matrix)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# ax1.plot(trajectory[:1000], color='blue', alpha=0.6)
# ax1.set_title("Continuous Trajectory (First 1000 steps showing metastability)")
# ax1.set_ylabel("Position (x)")

# ax2.plot(discrete_trajectory[:1000], color='red', linestyle='--')
# ax2.set_title("Discretized Trajectory (State 0 vs State 1)")
# ax2.set_ylabel("Assigned State")
# ax2.set_xlabel("Time Steps")
# plt.tight_layout()

# plt.savefig('markov_model.png',dpi=300,bbox_inches='tight')