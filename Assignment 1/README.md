# Assignment 1
Author: Goving S Ashan(ME23B168) & Yash Purswani(ME22B214)

## Overview
This repository contains the solutions and experimental setups for Assignment 1. The assignment consists of various questions exploring Policy Iteration, Q-Learning, and SARSA algorithms across multiple configurations.

## Question 1
- Contains the code for the **Value Iteration** algorithm.

## Question 2
The experiments and grid searches are structured as follows:

### Part 2.a
- **Q-Learning**: Grid search over 50,000 episodes using:
  - Learning Rates (`lrs`): `[0.01, 0.05, 0.1, 0.2]`
  - Epsilon Values (`eps_values`): `[0.5, 0.8, 1.0]`
- **SARSA**: Grid search over 50,000 episodes using:
  - Learning Rates (`lrs`): `[0.01, 0.05, 0.1]`
  - Epsilon Values (`eps_values`): `[0.3, 0.5, 0.8, 1.0]`

### Part 2.b
- **Q-Learning**: Search using:
  - Learning Rates (`lrs`): `[0.01, 0.05, 0.1, 0.2]`
  - Decay Rates (`decays`): `[0.9, 0.95, 0.99, 0.995, 0.999]`
- **SARSA**: Search using:
  - Learning Rates (`lrs`): `[0.005, 0.01, 0.05, 0.1]`
  - Decay Rates (`decays`): `[0.9, 0.95, 0.99, 0.995, 0.999]`

### Additional Details
- Data points from these grid searches and experiment iterations are saved in their respective directories.
- The optimal parameters found were then run with **10 seeds** across **100,000 episodes** using:
  - `sarsa_best.py`
  - `q_learning_best.py`
- Plotting and comparison of these results are handled by `spread_comparison_plot.py`.

## Question 3
- **File**: `sarsa_q_learning_comparison.py`
- **Part 3.i (Online Performance)**: Trained over 50,000 episodes for both SARSA and Q-Learning using the following parameters:
  - Learning Rate (`lr`): `0.1`
  - Initial Epsilon (`eps_start`): `1.0`
  - Optimal Decay Rate (`decay`): `0.9`
  - Final Epsilon (`end_eps`): `0.1`
- **Part 3.ii (Offline Performance)**: Evaluation and comparison carried out globally across 100 episodes for **3 seeds**.

## Question 4
- **Files**: `sarsa_bin_search.py` and `q_learning_bin_search.py`
- Run for 50,000 episodes (`n_episodes = 50000`) across **3 seeds** with the following configuration:
  - Learning Rate (`lr`): `0.1`
  - Initial Epsilon (`start_eps`): `1.0`
  - Final Epsilon (`final_eps`): `0.1`
  - Exponential Decay Rate (`exp_decay_rate`): `0.9`
  - Number of Bins (`bins_list`): `[5, 10, 15, 20]`

## Question 6: 
This experiment investigates how modified reward functions and the scaling factor $\eta$ influence the convergence rate of a Q-Learning agent.

* **Scripts**: `q_episode_vs_length.py`, `q_learning_mod_reward_episode_length.py`
* **Training Episodes**: 100,000 per configuration
* **Deterministic Seed**: 42
* **Comparison Groups**:
    * **Baseline**: Q-Learning using the standard environment reward function
    * **Modified Reward**: Q-Learning utilizing a shaped reward function across various $\eta$ values: `{0.5, 1, 2, 5}`
* **Primary Metric**: Evolution of **Episode Length** over time to measure the speed and stability of policy convergence
