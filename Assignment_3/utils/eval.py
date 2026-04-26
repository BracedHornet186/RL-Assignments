def evaluate(agent, env, episodes=20):
    total = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0

        while not done:
            action = agent.act(obs, sample=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        total += ep_ret

    return total / episodes