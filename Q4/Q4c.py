import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

rhos = [1, 2, 4, 8]

plt.figure(figsize=(10, 6))

for rho in rhos:
    df = pd.read_csv(os.path.join( f"all_returns_rho_{rho}.csv"))

    # rows = seeds, cols = episodes
    mean_returns = df.mean(axis=0)

    lower = []
    upper = []

    # compute tolerance interval at each episode
    for t in range(df.shape[1]):
        vals = df.iloc[:, t]
        lower.append(np.percentile(vals, 5))
        upper.append(np.percentile(vals, 95))

    lower = np.array(lower)
    upper = np.array(upper)

    plt.plot(mean_returns, label=f"rho={rho}")
    plt.fill_between(range(len(mean_returns)), lower, upper, alpha=0.2)

plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Mean Performance with (α=0.05, β=0.9) Tolerance Intervals")
plt.legend()
plt.grid(True)
plt.savefig("tolerance_intervals.png", dpi=300)
plt.show()