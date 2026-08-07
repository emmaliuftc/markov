# markov

Contains code for trajectory simulation and using Hidden Markov Models to coupling and decode two trajectories.

- `generate_markov.py`: Original Markov matrix definition with double-well potential particle
- `covariance.py`: Generating coupled trajectories with covariance matrix
- `verlet.py`: Implements GJ-F modified Verlet for trajectory on top of covariance coupling & analysis (PCCA, spectral analysis, C-K test, HMM analysis, synthetic trajectory generation)  
- `coupling.py`: Generating coupled trajectories with spring force & analysis 
- `hmm_test.py`: Implements Hidden Markov Models with spring force trajectories
- `markov_test.py`: Simple Markov matrix experimentation
- `pixi.lock` `pixi.toml`: pixi environment configuration files# markov
