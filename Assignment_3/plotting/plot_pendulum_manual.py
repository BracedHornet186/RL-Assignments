import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_runs(theta, alpha):
    # Updated pattern to look inside the specific mode subfolder
    search_pattern = f"logs/manual_a{alpha}/*theta{theta}_seed*.csv"
    files = glob.glob(search_pattern)

    runs = []
    steps = None

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)

        if steps is None:
            steps = data[:, 0]   # actual steps from file

        runs.append(data[:, 1])  # returns

    if not runs:
        print(f"Warning: No data found for theta={theta}, alpha={alpha} (Searched: {search_pattern})")
        return None, None

    return steps, np.array(runs)

def plot_q5a():
    targets = [-60, 90, 120, -150]
    alphas = [0.01, 0.05, 0.2, 0.5]
    
    # Create a 2x2 grid for the 4 target angles
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, theta in enumerate(targets):
        ax = axes[idx]
        
        for alpha in alphas:
            steps, runs = load_runs(theta, alpha)
            
            if steps is None:
                continue
            
            # Skip step 0 to avoid massive initial negative spike breaking the y-axis scale
            steps = steps[1:]
            runs = runs[:, 1:]
            
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            ci = std / np.sqrt(runs.shape[0])  # standard error
            
            # Plot the line and confidence interval
            line, = ax.plot(steps, mean, label=f"α = {alpha}", linewidth=2)
            color = line.get_color()
            ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)
            
        ax.set_title(f"Target Angle: θ = {theta}°", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Timesteps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Place legend in the bottom right corner
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        
    plt.suptitle("Q5(a): Effect of Fixed Entropy Temperature (α) on Different Targets", fontsize=18, y=1.02)
    plt.tight_layout()
    
    # Ensure plots directory exists and save
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", "q5a_manual_alpha_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Success! Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_q5a()