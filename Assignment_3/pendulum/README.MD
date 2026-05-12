# Pendulum Project

This project implements a Soft Actor-Critic (SAC) agent to solve the Pendulum environment.

## Folder Structure

The project is organized as follows:

```
.
├── agent/              # Contains the core SAC agent implementation
│   ├── actor.py        # Defines the actor network
│   ├── critic.py       # Defines the critic network
│   └── sac.py          # Implements the SAC algorithm
├── envs/               # Contains the custom pendulum environment
│   └── pendulum_target.py # Defines the pendulum environment with a target
├── logs/               # Stores the training logs
├── models/             # Stores the trained models
├── plotting/           # Contains scripts for plotting the results
├── runs/               # Contains scripts for running the experiments
├── utils/              # Contains utility functions for evaluation, logging, and replay buffer
├── train_pendulum.py   # The main script to train the SAC agent
└── README.MD           # This file
```

## How to Run

To run the code, you must be in the `pendulum` directory.

```bash
python train_pendulum.py
```
