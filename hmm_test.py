import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from hmmlearn import hmm

# STEP 1: Generate Coupled Trajectories

np.random.seed(42)
""" n_steps = 50000
dt = 0.01
noise_scaler = 0.4
 """

def generate_coupled_data(gamma, n_steps=50000, dt=0.01, noise=0.4):
    """Simulates the spring-coupled double-well particles."""
    x, y = -1.0, -1.0
    traj_X, traj_Y = np.zeros(n_steps), np.zeros(n_steps)
    
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
        x += force_x * dt + np.random.normal(0, noise)
        y += force_y * dt + np.random.normal(0, noise)
        
        traj_X[t] = x
        traj_Y[t] = y
        
    return traj_X, traj_Y

# ==========================================
# 2. Setup the Sweep Parameters
# ==========================================
gamma_values = np.linspace(0.0, 2.0, 40) # Test 20 points between 0 and 10.0
metastability_scores = [] # To track the diagonals of the transition matrix
emission_scores = []      # To track the sharpening of the emissions

print("Starting Gamma Sweep...")

for g in gamma_values:
    # A. Generate the data for this Gamma
    traj_X, traj_Y = generate_coupled_data(gamma=g)
    
    # B. Discretize (K-Means)
    discrete_X = KMeans(n_clusters=2, random_state=42, n_init='auto').fit_predict(traj_X.reshape(-1, 1))
    discrete_Y = KMeans(n_clusters=2, random_state=42, n_init='auto').fit_predict(traj_Y.reshape(-1, 1))
    
    # C. Create Joint States (0, 1, 2, 3)
    joint_discrete = discrete_X * 2 + discrete_Y
    
    # THE FIX: Subsample the data using a lag time (tau)
    tau = 20
    X_observed = joint_discrete[::tau].reshape(-1, 1)

    count_matrix_joint = np.zeros((4, 4))
    for i in range(len(joint_discrete) - tau):
        count_matrix_joint[joint_discrete[i], joint_discrete[i + tau]] += 1

    # Normalize rows to sum to 1 (add small epsilon to avoid divide-by-zero)
    empirical_matrix = count_matrix_joint / (count_matrix_joint.sum(axis=1, keepdims=True) + 1e-9)
    # X_observed = count_matrix_joint

    # D. Fit the HMM
    model = hmm.CategoricalHMM(n_components=2, random_state=42, n_iter=200)
    model.fit(X_observed)
    
    trans_mat = model.transmat_
    emiss_mat = model.emissionprob_
    
    # E. Fix "Label Switching"
    # We want Hidden State 0 to be the one that strongly emits Joint State 0.
    # If the HMM randomly assigned it backward, we flip the rows.
    if emiss_mat[1, 0] > emiss_mat[0, 0]:
        trans_mat = trans_mat[[1, 0], :][:, [1, 0]] # Swap rows and columns
        emiss_mat = emiss_mat[[1, 0], :]            # Swap rows
        
    # F. Record the Metrics
    # Metric 1: "Deepening" -> Average probability of staying in the same hidden state
    mean_diagonal = (trans_mat[0, 0] + trans_mat[1, 1]) / 2.0
    metastability_scores.append(mean_diagonal)
    
    # Metric 2: "Sharpening" -> Probability that Hidden State 0 emits exactly Joint State 0
    match_probability = emiss_mat[0, 0]
    emission_scores.append(match_probability)
    
    print(f"Gamma: {g:.2f} | Latent Stability: {mean_diagonal:.3f} | Emission Match: {match_probability:.3f}")

# ==========================================
# 3. Visualize the Results
# ==========================================
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Emission Sharpening (Red Line)
ax1.set_xlabel('Coupling Strength (Gamma)')
ax1.set_ylabel('Emission "Match" Probability', color='tab:red')
ax1.plot(gamma_values, emission_scores, color='tab:red', marker='o', linewidth=2, label="Emission Sharpening")
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0, 1.05)

# Create a second y-axis for the Transition Deepening (Blue Line)
ax2 = ax1.twinx()  
ax2.set_ylabel('Latent Metastability (Diagonal Mean)', color='tab:blue')
ax2.plot(gamma_values, metastability_scores, color='tab:blue', marker='s', linewidth=2, linestyle='--', label="Transition Deepening")
ax2.tick_params(axis='y', labelcolor='tab:blue')
# ax2.set_ylim(0.7, 1.0) # Zoomed in to show the subtle deepening

plt.title("HMM Matrix Evolution vs. Physical Coupling")
fig.tight_layout()
plt.grid(True, alpha=0.3)

name = input("figname: ")
plt.savefig(f"{name}.png",dpi=300,bbox_inches='tight')
plt.close()
