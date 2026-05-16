import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_runs(theta):
    # Ensure this matches your directory structure
    files = glob.glob(f"logs/auto/pendulum_theta{theta}_seed*.csv")

    runs = []
    steps = None

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)

        if steps is None:
            # Shift the steps so that 10000 becomes 0, 20000 becomes 10000, etc.
            steps = data[:, 0] - 10000   
            
        runs.append(data[:, 1])  # returns
    
    if steps is None:
        return None, None

    return steps, np.array(runs)

def plot_all():
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    targets = [0, -10, 30, -60, 90, -90, 120, -150]

    plt.figure(figsize=(10, 6))

    for theta in targets:
        steps, runs = load_runs(theta)

        if steps is None:
            print(f"Warning: No data found for theta={theta}. Skipping...")
            continue

        # --- Skip the first evaluation point (Now mapped to Step 0) ---
        # Fixed slice: [1:] actually removes the massive initial negative reward 
        # so the plot scales properly.
        steps = steps[:]
        runs = runs[:, :]

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
    
    # Force the X-axis to show ticks every 10,000 steps
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
    ax.set_xlim(left=0)
    # Place the legend cleanly in the bottom right corner
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Saves a high-res image for your assignment submission
    plt.savefig(os.path.join(save_dir, "sac_pendulum_plot.png"), dpi=300) 
    plt.show()

if __name__ == "__main__":
    plot_all()