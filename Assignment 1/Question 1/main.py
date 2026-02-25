import numpy as np

class DroneGridWorld:
    def __init__(self):
        self.grid_size = 5
        self.states = [(x, y, w) for x in range(5) for y in range(5) for w in [0, 1]] 
        self.actions = ['N', 'S', 'E', 'W', 'Hover']
        self.hazards = {(1,2), (3,2)}
        self.boulders = {(2, 4), (3, 4)}
        self.lake = (0, 0)
        self.fire_zone = (4, 4)

    def get_transitions(self, state, action):
        '''
        returns list of (next_state, probability, reward)
        '''
        x, y, has_water = state

        # 1. Handle Terminal States: If already at a boulder or success point, no further transitions [cite: 28, 29]
        if (x, y) in self.boulders or ((x, y) == self.fire_zone and has_water):
            return []

        # 2. Hover Action: Deterministic and unaffected by wind [cite: 25]
        if action == 'Hover':
            # Reward is -1 step penalty 
            return [(state, 1.0, -1)]

        # 3. Define Movement Probabilities [cite: 22, 23]
        if (x, y) in self.hazards:
            # Smoke fumes: Intended (40%), Stay (40%), Perpendiculars (10% each) [cite: 23]
            probs = {'intended': 0.4, 'stay': 0.4, 'perp1': 0.1, 'perp2': 0.1}
        else:
            # Normal cells: Intended (70%), Stay (10%), Perpendiculars (10% each) 
            probs = {'intended': 0.7, 'stay': 0.1, 'perp1': 0.1, 'perp2': 0.1}

        # 4. Map actions to direction vectors
        move_map = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
        perp_map = {
            'N': ['E', 'W'], 'S': ['E', 'W'],
            'E': ['N', 'S'], 'W': ['N', 'S']
        }

        possible_outcomes = []
        
        # Calculate movements for each stochastic outcome
        move_types = [
            (action, probs['intended']),
            (None, probs['stay']),
            (perp_map[action][0], probs['perp1']),
            (perp_map[action][1], probs['perp2'])
        ]

        for move_dir, prob in move_types:
            if prob == 0: continue
            
            # Calculate new coordinates
            if move_dir is None:
                nx, ny = x, y
            else:
                dx, dy = move_map[move_dir]
                nx, ny = x + dx, y + dy
                
                # Off-grid constraint: stay in same cell 
                if not (0 <= nx < 5 and 0 <= ny < 5):
                    nx, ny = x, y

            # 5. Determine New State and Phase Trigger [cite: 20]
            # Entering lake always fills water [cite: 20]
            new_has_water = 1 if (nx, ny) == self.lake else has_water
            next_state = (nx, ny, new_has_water)

            # 6. Determine Reward [cite: 28, 29]
            reward = -1  # Base per-step penalty 
            
            if (nx, ny) in self.hazards:
                reward -= 10  # Additional hazard penalty 
            elif (nx, ny) in self.boulders:
                reward = -100  # Crash cost 
            elif (nx, ny) == self.fire_zone and new_has_water:
                reward = 100  # Success payoff [cite: 29]

            possible_outcomes.append((next_state, prob, reward))

        return possible_outcomes


    def value_iteration(self, gamma=0.95, theta=1e-6):
        V = {state: 0 for state in self.states}

        while True:
            delta = 0
            for s in self.states:
                v_old = V[s]
                if self.terminal_check(s): 
                    continue
                
                action_values = []
                for a in self.actions:
                    expected_return =  sum(p*(r+gamma*V[ns]) for ns, p, r in self.get_transitions(s, a))
                    action_values.append(expected_return)
                
                V[s] = max(action_values)
                delta = max(delta, abs(V[s] - v_old))

            if delta < theta : break

        policy = {}
        for s in self.states:
            if self.terminal_check(s):
                continue
                
            action_returns = {}
            for a in self.actions:
                expected_return = sum(p * (r + gamma * V[ns]) 
                                    for ns, p, r in self.get_transitions(s, a))
                action_returns[a] = expected_return
            
            # Pythonic Argmax: find the key with the maximum value
            policy[s] = max(action_returns, key=action_returns.get)

        return V, policy




    def terminal_check(self, state):
        x, y, has_water = state
        if (x, y) in self.boulders or ((x, y) == self.fire_zone and has_water):
            return True
        else: return False

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_results(V, policy):
    phases = [0, 1]  # 0: Empty, 1: Filled
    phase_names = ["Phase 1: To Lake (No Water)", "Phase 2: To Fire (Has Water)"]
    
    # Action to Arrow mapping
    arrows = {'N': '↑', 'S': '↓', 'E': '→', 'W': '←', 'Hover': 'H', 'Done': 'X'}

    for w in phases:
        # 1. Prepare 2D arrays for the current phase
        v_grid = np.zeros((5, 5))
        p_grid = np.empty((5, 5), dtype=object)
        
        for x in range(5):
            for y in range(5):
                state = (x, y, w)
                v_grid[4-y, x] = V.get(state, 0) # 4-y to flip for grid display
                
                # Get policy arrow or terminal marker
                action = policy.get(state, 'Done')
                p_grid[4-y, x] = arrows.get(action, 'X')

        # 2. Create the Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(phase_names[w], fontsize=16)

        # Plot Value Function Heatmap
        sns.heatmap(v_grid, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax1, cbar=False)
        ax1.set_title("Value Function $V(s)$")

        # Plot Policy Arrows
        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 5)
        ax2.set_xticks(range(6))
        ax2.set_yticks(range(6))
        ax2.grid(True)
        ax2.set_title("Optimal Policy $\pi(s)$")

        for i in range(5):
            for j in range(5):
                ax2.text(i + 0.5, 4 - j + 0.5, p_grid[j, i], 
                         ha='center', va='center', fontsize=20)

    plt.tight_layout()
    plt.show()

        

drone_env = DroneGridWorld()
V, policy = drone_env.value_iteration()
visualize_results(V, policy)