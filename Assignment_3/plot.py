import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_runs(theta):
    files = glob.glob(f"logs/pendulum_theta{theta}_seed*.csv")
    runs = []

    for f in files:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        runs.append(data[:, 1])  # returns only

    return np.array(runs)


def plot_all():
    targets = [0, -10, 30, -60, 90, -90, 120, -150]

    plt.figure(figsize=(10, 6))

    for theta in targets:
        runs = load_runs(theta)

        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        ci = std / np.sqrt(runs.shape[0])  # ✅ confidence interval

        steps = np.arange(10000, 10000 * (len(mean) + 1), 10000)

        plt.plot(steps, mean, label=f"{theta}")
        plt.fill_between(steps, mean - ci, mean + ci, alpha=0.2)

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Average Return")
    plt.title("Pendulum SAC Performance")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_all()