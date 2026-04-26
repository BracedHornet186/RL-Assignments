import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_runs(mode, scale):
    # Resolve the exact subfolder name based on your updated train.py logic
    if mode == "auto":
        folder_name = f"auto_rs{scale}"
    else:
        folder_name = f"{mode}_rs{scale}" # e.g., manual_a0.01_rs10.0
        
    # Construct explicit search paths (dropping the specific 'q5b_' prefix just in case)
    primary_search = f"logs/{folder_name}/*theta90_seed*_{mode}_rs{scale}.csv"
    files = glob.glob(primary_search)
    
    # Fallback 1: Check the root logs folder
    if not files: 
        files = glob.glob(f"logs/*theta90_seed*_{mode}_rs{scale}.csv")
        
    # Fallback 2: General recursive search
    if not files:
        files = glob.glob(f"logs/**/*theta90_seed*_{mode}_rs{scale}.csv", recursive=True)

    runs = []
    steps = None

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        data = np.atleast_2d(data)

        if steps is None:
            steps = data[:, 0]   # actual steps from file

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
            steps_m = steps_m[1:]
            runs_m = runs_m[:, 1:]
            
            mean_m = runs_m.mean(axis=0)
            ci_m = runs_m.std(axis=0) / np.sqrt(runs_m.shape[0])
            
            line_m, = ax.plot(steps_m, mean_m, label=f"Manual α = {best_alpha}", linewidth=2, color='red')
            ax.fill_between(steps_m, mean_m - ci_m, mean_m + ci_m, alpha=0.2, color='red')

        # 2. Load and plot AUTOMATED tuning data (Blue)
        steps_a, runs_a = load_runs("auto", scale)
        if steps_a is not None:
            steps_a = steps_a[1:]
            runs_a = runs_a[:, 1:]
            
            mean_a = runs_a.mean(axis=0)
            ci_a = runs_a.std(axis=0) / np.sqrt(runs_a.shape[0])
            
            line_a, = ax.plot(steps_a, mean_a, label="Auto-Tuned α", linewidth=2, color='blue')
            ax.fill_between(steps_a, mean_a - ci_a, mean_a + ci_a, alpha=0.2, color='blue')
            
        ax.set_title(f"Reward Scale: {scale}x", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Timesteps", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
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