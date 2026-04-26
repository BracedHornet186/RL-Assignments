# Assignment_3/plotting/plot_learning_curves.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_individual_learning_curve(reward_type, log_dir="logs/reacher", output_dir="plots"):
    """
    Plots the learning curve for SAC trained on reward_type and evaluated on the same reward_type.
    Includes horizontal lines and legend markers for the 'Best' and 'Last' mean returns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_pattern = os.path.join(log_dir, f"reacher_easy_{reward_type}_seed*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found for SAC-{reward_type} in {log_dir}. Skipping...")
        return
        
    all_data = []
    steps = None
    
    for f in files:
        df = pd.read_csv(f)
        
        if steps is None:
            steps = df['step'].values
            
        eval_col = f"return_{reward_type}"
        
        if eval_col in df.columns:
            all_data.append(df[eval_col].values)
            
    if all_data:
        # Handle potential length mismatches (if some seeds stopped early)
        min_len = min([len(d) for d in all_data])
        steps_trimmed = steps[:min_len]
        all_data_trimmed = np.array([d[:min_len] for d in all_data])
        
        # Calculate mean and standard deviation for confidence interval
        mean_returns = np.mean(all_data_trimmed, axis=0)
        std_returns = np.std(all_data_trimmed, axis=0)
        
        # Calculate Best and Last from the mean curve
        best_return = np.max(mean_returns)
        last_return = mean_returns[-1]
        
        plt.figure(figsize=(8, 5))
        
        # Define specific base colors
        color = {"Ra": "blue", "Rb": "orange", "Rc": "green"}[reward_type]
        
        # Plot main learning curve and shaded area
        plt.plot(steps_trimmed, mean_returns, label=f"SAC-$\mathcal{{R}}_{reward_type[-1]}$ Mean", color=color, linewidth=2)
        plt.fill_between(
            steps_trimmed, 
            mean_returns - std_returns, 
            mean_returns + std_returns, 
            alpha=0.2, 
            color=color
        )
        
        # Add Best and Last horizontal lines
        plt.axhline(y=best_return, color='red', linestyle='--', alpha=0.8, 
                    label=f"Best: {best_return:.1f}")
        plt.axhline(y=last_return, color='purple', linestyle=':', alpha=0.8, 
                    label=f"Last: {last_return:.1f}")
        
        # Format the plot
        plt.title(f"Learning Curve: SAC-$\mathcal{{R}}_{reward_type[-1]}$ Evaluated on $\mathcal{{R}}_{reward_type[-1]}$", fontsize=14)
        plt.xlabel("Environment Timesteps", fontsize=12)
        plt.ylabel("Average Undiscounted Return", fontsize=12)
        
        # Place legend nicely
        plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Save and show
        save_path = os.path.join(output_dir, f"learning_curve_{reward_type}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Check if the path needs adjusting if run from inside plotting/ dir
    base_dir = "logs/reacher"
    if not os.path.exists(base_dir):
        base_dir = "../logs/reacher"
        out_dir = "../plots"
    else:
        out_dir = "plots"

    # Generate the 3 separate plots asked for in Question 2.3.2
    for r_type in ["Ra", "Rb", "Rc"]:
        plot_individual_learning_curve(r_type, log_dir=base_dir, output_dir=out_dir)