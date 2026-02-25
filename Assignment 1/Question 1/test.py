"""
Let's use Value Iteration to solve FrozenLake!

Setup
-----
We start off by defining our actions:
A = {move left, move right...} = {(0,1),(0,-1),...}
S = {(i,j) for 0 <= i,j < 4}
Reward for (3,3) = 1, and otherwise 0.
Probability distribution is a 4x(4x4) matrix of exactly the policy.
We have pi(a|s), where a in A, and s in S.

Problem formulation : https://gym.openai.com/envs/FrozenLake-v0/

Algorithm
---------
Because our situation is deterministic for now, we have the value iteration eq:

v <- 0 for all states.
v_{k+1}(s) = max_a (\sum_{s',r} p(s',r|s,a) (r + \gamma * v_k(s'))

... which decays to:

v_{k+1}(s = max_a (\sum_{s'} 1_(end(s')) + \gamma * v_k(s'))

Because of our deterministic state and the deterministic reward.
"""
import numpy as np
import matplotlib.pyplot as plt
N = 4
v = np.zeros((N, N), dtype=np.float32) # Is our value vector.
THRESHOLD = 1e-5
A = [(0,1),(0,-1),(1,0),(-1,0)]
MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

def value_iter(V_prev):
    V_next = R + P * V_prev
    while V_next != V_prev:
        V_next = R + P * V_prev
    return V_next

def policy_improvement(prev_pi):
    for s in S:
    	pi(s) = prev_pi(s)
    for s in S:
    	pi(s) = argmax_a(q(s,a))
    return pi

def policy_iter(pi_prev):
    # Initialize
    q = find_q(pi_prev)
    V = find_V(pi_prev)
    pi_next = policy_improvement(pi_prev)

    # Policy Iteration
    while pi_next != pi_prev:
        pi_prev = pi_next
        q = find_q(pi_prev)
        V = find_V(pi_prev)
        pi_next = policy_improvement(pi_prev)
    return pi_next
    
def value_iter(prev_V):
    V = None
    while prev_V != V:
        V = prev_V
        for s in S:
            max_v_s = 0
            for a in A:
                # P(s,a) is a |S|x|A| matrix, where i,j=p(i,j|s,a)
                max_v_s = max(P(s,a) * (R(s,a) + gamma V[s]), max_v_s)
            V[s] = max_v_s
    return V # returns optimal V

def proj(n, minn, maxn):
    """
    projects n into the range [minn, maxn). 
    """
    return max(min(maxn-1, n), minn)

def move(s, tpl, stochasticity=0):
    """
    Set stochasticity to any number in [0,1].
    This is equivalent to "slipping on the ground"
    in FrozenLake.
    """
    if MAP[s[0]][s[1]] == 'H': # Go back to the start
        return (0,0)
    if np.random.random() < stochasticity:
        return random.choice(A)
    return (proj(s[0] + tpl[0], 0, N), proj(s[1] + tpl[1], 0, N))

def reward(s):
    return MAP[s[0]][s[1]] == 'G'
    
def run_with_value(v, gamma=0.9):
    old_v = v.copy()
    for i in range(N):
        for j in range(N):
            best_val = 0
            for a in A:
                new_s = move((i,j), a)
                best_val = max(best_val, reward(new_s) + gamma * old_v[new_s])
            v[i,j] = best_val
    return old_v

# Performing Value Iteration
plt.matshow(v)
old_v = run_with_value(v)
while norm(v - old_v) >= THRESHOLD:
    old_v = run_with_value(v)
plt.matshow(v)

# Extracting policy from v:
def pi(s, v):
    cur_best = float("-inf")
    cur_a = None
    for a in A:
        val = v[move(s, a)]
        if val > cur_best:
            cur_a = a
            cur_best = val
    return cur_a

# Plotting a nice arrow map.
action_map = np.array([
    [pi((i,j), v) for j in range(N)] for i in range(N)])
Fx = np.flip(np.array([ [col[1] for col in row] for row in action_map ]),0)
Fy = np.flip([ [-col[0] for col in row] for row in action_map ],0)
plt.quiver(Fx,Fy)
