import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_runs(mode, scale):
    # Resolve the exact subfolder name based on your updated train.py logic
    if mode == "auto":
        folder_name = f"auto_rs{scale}"
    else:
        folder_name = f"{mode}_rs{scale}" # e.g., manual_a0.01_rs10.0
        
    # Construct explicit search paths (dropping the specific 'q5b_' prefix just in case)
    primary_search = f"logs/{folder_name}/*theta90_seed*_{mode}_rs{scale}.csv"
    files = glob.glob(primary_search)

    runs = []
    steps = None

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)

        if steps is None:
            # Shift the steps so that 10000 becomes 0
            steps = data[:, 0] - 10000   

        runs.append(data[:, 1])  # returns

    if not runs:
        print(f"Warning: No data found for mode={mode}, scale={scale}x")
        print(f"  -> Searched primary path: {primary_search}")
        return None, None

    return steps, np.array(runs)

def plot_q5b():
    target = 90
    best_alpha = 0.01  # Updated to match your chosen best manual alpha
    scales = [10.0, 0.1]
    
    # Create a 1x2 grid for the two reward scales
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, scale in enumerate(scales):
        ax = axes[idx]
        
        # 1. Load and plot MANUAL tuning data (Red)
        steps_m, runs_m = load_runs(f"manual_a{best_alpha}", scale)
        if steps_m is not None:
            steps_m = steps_m[:]
            runs_m = runs_m[:, :]
            
            mean_m = runs_m.mean(axis=0)
            ci_m = runs_m.std(axis=0) / np.sqrt(runs_m.shape[0])
            best_idx_m = np.argmax(mean_m)
            best_step_m = steps_m[best_idx_m]
            best_value_m = mean_m[best_idx_m]
            last_step_m = steps_m[-1]
            last_value_m = mean_m[-1]
            
            label_m = (
                f"Manual α = {best_alpha} | "
                f"Best: {best_value_m:.1f} | Last: {last_value_m:.1f}"
            )
            line_m, = ax.plot(steps_m, mean_m, label=label_m, linewidth=2, color='red')
            ax.fill_between(steps_m, mean_m - ci_m, mean_m + ci_m, alpha=0.2, color='red')
            ax.scatter(best_step_m, best_value_m, color='red', marker="o", s=55, zorder=5)
            ax.scatter(last_step_m, last_value_m, color='red', marker="s", s=55, zorder=5)

        # 2. Load and plot AUTOMATED tuning data (Blue)
        steps_a, runs_a = load_runs("auto", scale)
        if steps_a is not None:
            steps_a = steps_a[1:]
            runs_a = runs_a[:, 1:]
            
            mean_a = runs_a.mean(axis=0)
            ci_a = runs_a.std(axis=0) / np.sqrt(runs_a.shape[0])
            best_idx_a = np.argmax(mean_a)
            best_step_a = steps_a[best_idx_a]
            best_value_a = mean_a[best_idx_a]
            last_step_a = steps_a[-1]
            last_value_a = mean_a[-1]
            
            label_a = (
                f"Auto-Tuned α | "
                f"Best: {best_value_a:.1f} | Last: {last_value_a:.1f}"
            )
            line_a, = ax.plot(steps_a, mean_a, label=label_a, linewidth=2, color='blue')
            ax.fill_between(steps_a, mean_a - ci_a, mean_a + ci_a, alpha=0.2, color='blue')
            ax.scatter(best_step_a, best_value_a, color='blue', marker="o", s=55, zorder=5)
            ax.scatter(last_step_a, last_value_a, color='blue', marker="s", s=55, zorder=5)
            
        ax.set_title(f"Reward Scale: {scale}x", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Timesteps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force the X-axis to show ticks every 10,000 steps
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
        ax.set_xlim(left=0)
        ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
        
    plt.suptitle(f"Q5(b): Manual vs Automated Temperature Tuning (θ = 90°)", fontsize=18, y=0.95)
    plt.tight_layout()
    
    # Ensure plots directory exists and save
    os.makedirs("plots", exist_ok=True)
    save_path = os.path.join("plots", "q5b_reward_scaling_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Success! Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_q5b()