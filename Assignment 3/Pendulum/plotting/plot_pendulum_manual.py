import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
            steps = data[:, 0] - 10000   # actual steps from file shifted to start at 0

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
            steps = steps[:]
            runs = runs[:, :]
            
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            ci = std / np.sqrt(runs.shape[0])  # standard error
            
            best_idx = np.argmax(mean)
            best_step = steps[best_idx]
            best_value = mean[best_idx]
            last_step = steps[-1]
            last_value = mean[-1]

            label = (
                f"α = {alpha} | "
                f"Best: {best_value:.1f} | Last: {last_value:.1f}"
            )

            # Plot the line, confidence interval, and best/last markers
            line, = ax.plot(steps, mean, label=label, linewidth=2)
            color = line.get_color()
            ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)
            ax.scatter(best_step, best_value, color=color, marker="o", s=45, zorder=5)
            ax.scatter(last_step, last_value, color=color, marker="s", s=45, zorder=5)
            
        ax.set_title(f"Target Angle: θ = {theta}°", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Timesteps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force the X-axis to show ticks every 10,000 steps for each subplot
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.set_xlim(left=0)
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