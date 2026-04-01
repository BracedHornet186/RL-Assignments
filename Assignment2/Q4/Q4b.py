import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

rhos = [1, 2, 4, 8]
perf_dict = {}

# =========================
# Extract performance per seed
# =========================
for rho in rhos:
    path = os.path.join(f"all_returns_rho_{rho}.csv")
    df = pd.read_csv(path)
    print(df.shape)
    # Each column = one seed
    final_perf = df.iloc[:, -50:].mean(axis=1)
    perf_dict[rho] = final_perf.values

# =========================
# 1. KDE Plot (Distribution)
# =========================
plt.figure(figsize=(10, 6))

for rho in rhos:
    sns.kdeplot(perf_dict[rho], label=f"rho={rho}", fill=True, alpha=0.3)

plt.title("Distribution of Final Performance (Last 50 Episodes)")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig("distribution_kde.png", dpi=300)
plt.show()

# =========================
# 2. Styled Box Plot
# =========================
plt.figure(figsize=(9, 6))

data = [perf_dict[rho] for rho in rhos]

box = plt.boxplot(
    data,
    labels=[f"ρ={r}" for r in rhos],
    patch_artist=True,   # allows coloring
    # showmeans=True,
    # meanline=True,
    widths=0.5
)

# -----------------------
# Styling
# -----------------------
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')

# Median line
for median in box['medians']:
    median.set(color='black', linewidth=2)

# Mean line
for mean in box['means']:
    mean.set(color='red', linewidth=2, linestyle='--')

# Whiskers
for whisker in box['whiskers']:
    whisker.set(color='black', linewidth=1.5)

# Caps
for cap in box['caps']:
    cap.set(color='black', linewidth=1.5)

# Outliers
for flier in box['fliers']:
    flier.set(marker='o', color='black', alpha=0.5)

# -----------------------
# Labels & Title
# -----------------------
plt.title("Final Performance Distribution Across Replay Factors", fontsize=14, weight='bold')
plt.xlabel("Replay Factor (ρ)", fontsize=12)
plt.ylabel("Return (Last 50 Episodes Mean)", fontsize=12)

# Grid & background
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.gca().set_facecolor('#f7f7f7')

plt.tight_layout()
plt.savefig("distribution_box.png", dpi=300)
plt.show()

# =========================
# 3. Violin Plot
# =========================
plt.figure(figsize=(8, 6))

all_data = []
labels = []

for rho in rhos:
    all_data.extend(perf_dict[rho])
    labels.extend([f"rho={rho}"] * len(perf_dict[rho]))

df_plot = pd.DataFrame({
    "Performance": all_data,
    "rho": labels
})

sns.violinplot(data=df_plot, x="rho", y="Performance")

plt.title("Violin Plot of Final Performance")
plt.grid(True)
plt.savefig("distribution_violin.png", dpi=300)
plt.show()