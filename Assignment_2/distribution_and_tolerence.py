import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling
sns.set_theme(style="whitegrid")

# Configuration for Tolerance Intervals (Question 4c)
ALPHA = 0.05  # 95% confidence that the interval contains at least Beta of population
BETA = 0.9    # Coverage (90% of the population)

def get_k_factor(n, alpha=ALPHA, beta=BETA):
    """Calculates the k-factor for a normal tolerance interval."""
    df = n - 1
    # Critical value of chi-square with df and alpha
    chi2_crit = stats.chi2.ppf(alpha, df)
    # Z-score for the desired coverage beta
    z_beta = stats.norm.ppf((1 + beta) / 2)
    # Howe's approximation or exact-like form for k
    k = z_beta * np.sqrt((df * (1 + 1/n)) / chi2_crit)
    return k

# 1. Load Data
log_pattern = re.compile(r"trunc2000_rho(\d+)_seed(\d+)\.csv")
all_runs = []

for file in os.listdir("."):
    match = log_pattern.match(file)
    if match:
        rho = int(match.group(1))
        seed = int(match.group(2))
        df = pd.read_csv(file)
        df['rho'] = rho
        df['seed'] = seed
        all_runs.append(df)

df_all = pd.concat(all_runs, ignore_index=True)

# --- 4(b) Plotting: Distribution of Performance ---
# Aggregate Performance is the mean return of a single run [cite: 100, 112]
agg_perf = df_all.groupby(['rho', 'seed'])['return'].mean().reset_index()

plt.figure(figsize=(10, 6))
for rho in sorted(agg_perf['rho'].unique()):
    subset = agg_perf[agg_perf['rho'] == rho]['return']
    sns.kdeplot(subset, label=f'$\\rho={rho}$', fill=True, alpha=0.3)

plt.title("4(b) Distribution of Aggregate Performance per Replay Factor")
plt.xlabel("Aggregate Performance (Mean Return)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("plot_4b_distribution.png")

# --- 4(c) Plotting: Tolerance Intervals ---
# Group by rho and episode to get statistics across seeds
stats_df = df_all.groupby(['rho', 'episode'])['return'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(12, 8))
colors = sns.color_palette("tab10", len(stats_df['rho'].unique()))

for i, rho in enumerate(sorted(stats_df['rho'].unique())):
    subset = stats_df[stats_df['rho'] == rho].sort_values('episode')
    n = subset['count'].iloc[0] 
    k = get_k_factor(n)
    
    # Calculate Tolerance Interval bounds
    lower = subset['mean'] - k * subset['std']
    upper = subset['mean'] + k * subset['std']
    
    plt.plot(subset['episode'], subset['mean'], label=f'$\\rho={rho}$ (Mean)', color=colors[i])
    plt.fill_between(subset['episode'], lower, upper, color=colors[i], alpha=0.15, 
                     label=f'$\\rho={rho}$ ($90\\%$ Tol. Int.)')

plt.title(f"4(c) Learning Curves with $(\\alpha={ALPHA}, \\beta={BETA})$ Tolerance Intervals")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plot_4c_tolerance.png")

print("Plots saved: plot_4b_distribution.png and plot_4c_tolerance.png")