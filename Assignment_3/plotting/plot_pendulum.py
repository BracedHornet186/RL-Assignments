import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_runs(theta):
    files = glob.glob(f"logs/pendulum_theta{theta}_seed*.csv")

    runs = []
    steps = None

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)

        if steps is None:
            steps = data[:, 0]   # actual steps from file

        runs.append(data[:, 1])  # returns

    return steps, np.array(runs)

def plot_all():
    cwd = os.getcwd()
    save_dir = os.path.join(cwd,'plots')
    targets = [0,-10,30, -60, 90, -90, 120, -150]

    plt.figure(figsize=(10, 6))

    for theta in targets:
        steps, runs = load_runs(theta)

        # --- Skip the first evaluation point (Step 0) ---
        # This removes the massive initial negative reward from the plot scale
        steps = steps[1:]
        
        runs = runs[:, 1:]

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        ci = std / np.sqrt(runs.shape[0])

        # Get best and last values
        best_idx = np.argmax(mean)
        best_step = steps[best_idx]
        best_value = mean[best_idx]

        last_step = steps[-1]
        last_value = mean[-1]

        # Format the legend label to include Best and Last cleanly
        label_text = f"θ={theta} (Best: {best_value:.1f} | Last: {last_value:.1f})"

        # Plot line and grab its color
        line, = plt.plot(steps, mean, label=label_text, linewidth=2)
        color = line.get_color()

        # Plot confidence interval
        plt.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)

        # Plot markers for Best (circle) and Last (square) without floating text
        plt.scatter(best_step, best_value, color=color, marker='o', s=60, zorder=5)
        plt.scatter(last_step, last_value, color=color, marker='s', s=60, zorder=5)

    plt.xlabel("Environment Timesteps", fontsize=12)
    plt.ylabel("Average Undiscounted Return", fontsize=12)
    plt.title("SAC Performance: Pendulum-v1", fontsize=14)
    
    # Place the legend cleanly in the bottom right corner
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Saves a high-res image for your assignment submission
    plt.savefig(os.path.join(save_dir,"sac_pendulum_plot.png"), dpi=300) 
    plt.show()

if __name__ == "__main__":
    plot_all()