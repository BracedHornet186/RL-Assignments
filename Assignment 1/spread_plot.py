import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_results(folder_path, window_size=100):
    # 1. Load all seed files
    files = [f for f in os.listdir(folder_path) if f.startswith("run_seed_") and f.endswith(".csv")]
    
    if not files:
        print(f"No files found in {folder_path}")
        return

    all_runs = []

    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        # Apply moving average to smooth the noise for this specific seed
        smoothed_reward = df["reward"].rolling(window=window_size).mean()
        all_runs.append(smoothed_reward)

    # 2. Aggregate smooth reward data (Rows = Episodes, Columns = Seeds)
    data_matrix = pd.concat(all_runs, axis=1)
    
    # Calculate statistics across the seeds (axis=1)
    mean_curve = data_matrix.mean(axis=1)
    std_curve = data_matrix.std(axis=1)
    
    # 3. Create the Plot
    plt.style.use('seaborn-v0_8-darkgrid') # Optional: cleaner look
    plt.figure(figsize=(12, 6))
    
    episodes = np.arange(len(mean_curve))
    
    # Plot the shaded variance (Mean +/- 1 Standard Deviation)
    plt.fill_between(
        episodes, 
        mean_curve - std_curve, 
        mean_curve + std_curve, 
        color='blue', 
        alpha=0.2, 
        label='$\pm$ 1 Std. Dev.'
    )
    
    # Plot the bold mean line
    plt.plot(episodes, mean_curve, color='blue', lw=2, label='Mean Reward')
    
    plt.title(f"Acrobot-v1: Q-Learning Performance (Average of {len(files)} Seeds)", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(f"Reward ({window_size} Episode Moving Avg)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig("q_learning_variance_plot.png", dpi=500)
    print("Plot saved as q_learning_variance_plot.png")

if __name__ == "__main__":
    plot_training_results("q_learning_best", window_size=1000)