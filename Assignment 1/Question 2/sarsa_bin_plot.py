import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print("\nGenerating Spread Plot for num_bins Comparison...")
output_folder = os.path.join("Assignment 1", "Question 2", "sarsa_bin_search_data")
bins_list = [5, 10, 15, 20]
seeds = [42, 101, 202]
plt.figure(figsize=(12, 8))
window_size = 1000 # Smoothing window for the 100,000 episodes

for num_bins in bins_list:
    bin_data = []
    for seed in seeds:
        # Load the CSV for this specific seed and bin combination
        file_path = os.path.join(output_folder, f"run_seed_{seed}_bins_{num_bins}.csv")
        df = pd.read_csv(file_path)
        
        # Apply rolling mean to the individual seed's data
        smoothed = df['reward'].rolling(window=window_size, min_periods=1).mean().values
        bin_data.append(smoothed)
        
    # Stack into shape (num_seeds, num_episodes)
    bin_data = np.vstack(bin_data)
    
    # Calculate mean and std across the 3 seeds
    mean_rewards = np.mean(bin_data, axis=0)
    std_rewards = np.std(bin_data, axis=0)
    
    episodes = np.arange(len(mean_rewards))
    
    # Plot the mean line and grab its color
    line, = plt.plot(episodes, mean_rewards, label=f"{num_bins} Bins (Mean)")
    
    # Plot the shaded region for +/- 1 standard deviation
    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color=line.get_color(),
        alpha=0.15, # 15% opacity so overlapping spreads are readable
    )

plt.title(f"SARSA Convergence by State Discretization (Rolling Avg: {window_size} eps)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(loc='lower right')
plt.grid(True)

# Save the plot
plot_path = os.path.join(output_folder, "bins_comparison_spread_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Spread plot successfully saved to: {plot_path}")