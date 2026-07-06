import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


p_init = np.array([1/3., 1/3., 1/3.])

# State 0: 0.1 probability
# State 1: 0.8 probability
# State 2: 0.1 probability
p_init = np.array([0.1, 0.8, 0.1])


p_transition = np.array(
    [[0.90, 0.05, 0.05],
     [0.01, 0.90, 0.09],
     [0.07, 0.03, 0.9]]
)

p_transition_example = np.array(
    [[0.6,  0.2, 0.2],
     [0.05, 0.9, 0.05],
     [0.1,  0.2, 0.7]]
)

assert p_transition[0, :].sum() == 1
assert p_transition[1, :].sum() == 1
assert p_transition[2, :].sum() == 1

p_next = p_init @ p_transition_example

print(p_next)

p_state_t = [p_init]

for i in range(200):
    p_state_t.append(p_state_t[-1] @ p_transition_example)

state_distributions = pd.DataFrame(p_state_t)

plt.plot(state_distributions)

plt.savefig('vis.png',dpi=300,bbox_inches='tight')

# Find the equilibrium state given the transition matrix (doesn't matter what the original state is because it should always reach the equilibrium state)
# The equilibrium state is effectively equal to the eigenvector of the transition matrix corresponding to an eigenvalue of 1 (the largest possible)

def equilibrium_distribution(p_transition):
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1,-1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq

print(equilibrium_distribution(p_transition_example))