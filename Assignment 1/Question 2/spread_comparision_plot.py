import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison_results(folders, labels, colors, window_size=1000):
    plt.style.use('seaborn-v0_8-darkgrid') 
    plt.figure(figsize=(12, 6))
    
    # Iterate through both algorithms
    for folder_path, label, color in zip(folders, labels, colors):
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist. Skipping...")
            return
            
        files = [f for f in os.listdir(folder_path) if f.startswith("run_seed_") and f.endswith(".csv")]
        
        if not files:
            print(f"No valid CSV files found in {folder_path}. Skipping...")
            return

        all_runs = []

        for file in files:
            df = pd.read_csv(os.path.join(folder_path, file))
            # Apply moving average to smooth the noise for this specific seed
            smoothed_reward = df["reward"].rolling(window=window_size).mean()
            all_runs.append(smoothed_reward)

        # Aggregate smooth reward data (Rows = Episodes, Columns = Seeds)
        data_matrix = pd.concat(all_runs, axis=1)
        
        # Calculate statistics across the seeds (axis=1)
        mean_curve = data_matrix.mean(axis=1)
        std_curve = data_matrix.std(axis=1)
        
        episodes = np.arange(len(mean_curve))
        
        # Plot the shaded variance (Mean +/- 1 Standard Deviation)
        plt.fill_between(
            episodes, 
            mean_curve - std_curve, 
            mean_curve + std_curve, 
            color=color, 
            alpha=0.2, 
            label=f'{label} ($\pm$ 1 Std. Dev.)'
        )
        
        # Plot the bold mean line
        plt.plot(episodes, mean_curve, color=color, lw=2, label=f'{label} Mean')

    # Formatting the overall plot
    plt.title("Acrobot-v1: Q-Learning vs SARSA Performance (10 Seeds)", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel(f"Reward ({window_size} Episode Moving Avg)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = os.path.join("Assignment 1", "Question 2", "q_learning_vs_sarsa_plot.png")
    plt.savefig(output_filename, dpi=500)
    print(f"Plot successfully saved as {output_filename}")

if __name__ == "__main__":
    # Define the inputs for the comparison
    q_learning_folder = os.path.join("Assignment 1", "Question 2", "q_learning_best")
    sarsa_folder = os.path.join("Assignment 1", "Question 2", "sarsa_best")
    target_folders = [q_learning_folder, sarsa_folder]
    plot_labels = ["Q-Learning", "SARSA"]
    plot_colors = ["blue", "orange"]
    
    plot_comparison_results(
        folders=target_folders, 
        labels=plot_labels, 
        colors=plot_colors, 
        window_size=1000
    )