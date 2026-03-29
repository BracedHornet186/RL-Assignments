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
# 2. Box Plot
# =========================
plt.figure(figsize=(8, 6))

data = [perf_dict[rho] for rho in rhos]
plt.boxplot(data, labels=[f"rho={r}" for r in rhos])

plt.title("Box Plot of Final Performance")
plt.ylabel("Return")
plt.grid(True)
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