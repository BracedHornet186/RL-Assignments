import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_acrobot_results(file_path, window_size=10000):
    # 1. Load the data from CSV
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure the training script saved the file.")
        return

    # 2. Calculate Moving Averages
    # We use .rolling() from pandas which is cleaner for CSV data
    # df['q_smooth'] = df['q_learning'].rolling(window=window_size).mean()
    # df['sarsa_smooth'] = df['sarsa'].rolling(window=window_size).mean()

    df['reward_smoothed'] = df['reward'].rolling(window=window_size).mean()

    # 3. Create the Plot
    plt.figure(figsize=(12, 7))

    # Plot original raw data with high transparency (optional, for context)
    # plt.plot(df['q_learning'], color='blue', alpha=0.1, label='_nolegend_')
    # plt.plot(df['sarsa'], color='orange', alpha=0.1, label='_nolegend_')

    # Plot the smoothed lines
    # plt.plot(df['q_smooth'], color='blue', label=f'Q-Learning (MA {window_size})', linewidth=2)
    # plt.plot(df['sarsa_smooth'], color='orange', label=f'SARSA (MA {window_size})', linewidth=2)

    plt.plot(df['reward_smoothed'], color='orange', label=f'SARSA (MA {window_size})', linewidth=2)



    # 4. Formatting to meet assignment standards
    plt.title("Acrobot-v1 Learning Performance: Q-Learning vs SARSA", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Total Reward (Return)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the finalized figure
    plt.savefig("acrobot_final_plot.png", dpi=300)
    plt.show()

# Run the function
plot_acrobot_results("raw_lr_0.005_eps_1.0.csv", window_size=4000)