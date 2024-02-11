def greedyAgent(environment, maxEpisodes):
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))
    N = np.zeros((environment.observation_space.n, environment.action_space.n))
    R = np.zeros((environment.observation_space.n, environment.action_space.n))
    episode_pbar = tqdm(total=maxEpisodes, bar_format="{l_bar}{bar:20}{r_bar}")
    for episode in range(maxEpisodes):
        terminated = False
        truncated = False
        observation, info = environment.reset(seed=CONFIG['seed'])
        while not (terminated or truncated):
            action = np.argmax(Q[observation["agent"]])
            observation, reward, terminated, truncated, info = environment.step(action)
            N[observation["agent"]][action] += 1
            R[observation["agent"]][action] += reward
            # Q[observation["agent"]][action] = R[observation["agent"]][action] / N[observation["agent"]][action]
            Q[observation["agent"]][action] += (reward - Q[observation["agent"]][action]) / N[observation["agent"]][action]
            episode_pbar.set_postfix({'Q': Q[1], 'N': N[1], 'R': R[1]})
        episode_pbar.update(1)
    return Q