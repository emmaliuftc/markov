import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

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
        
        x += force_x * alpha * dt + noise_term
        
        beta_curr = beta_next

        traj_X[t] = x
        
    return traj_X

# ==========================================
# Run the Simulation
# ==========================================
mass = 1.0
drag_coeff = 0.001
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


plt.plot(euler_positions, label="Position (r)", color="blue")
plt.title(f"Euler Trajectory in a Double-Well Potential (alpha={drag_coeff},t={temperature},timestep={time_step})")

plt.savefig(f"euler_alpha{drag_coeff}.png",dpi=300,bbox_inches="tight")



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




