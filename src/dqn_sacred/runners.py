import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn_sacred.dqn import DQN_Agent


def test_agent(agent, env_id):
    test_env = gym.make(env_id)
    test_env.seed(123)
    state = test_env.reset()
    ep_rew = 0
    ep_len = 0
    done = False
    while not (done or ep_len == 200):
        action = agent.get_action(state, eps=0)
        new_state, reward, done, _ = test_env.step(action)
        state = new_state
        ep_rew += reward
        ep_len += 1
    return ep_rew, ep_len


def main_runner(n_episodes, seed, batch_size, alpha, gamma):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env_id = "CartPole-v1"
    # env_id = "LunarLander-v2"
    reward_thr = abs(gym.spec(env_id).reward_threshold)
    env = gym.make(env_id)
    env.seed(seed)
    agent = DQN_Agent(env, alpha=alpha, gamma=gamma)
    # noise = OUNoise(env.action_space)
    # noise_scale = 0.4
    rewards = []
    test_rewards = []
    test_avg = []
    avg_rewards = []
    q_losses = []
    loss = 0
    temp_factor = 1
    temp = 0.05
    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        episode_reward = 0
        for step in range(200):
            action = agent.get_action(state, eps=temp)
            new_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > batch_size:
                loss = agent.update(batch_size)
                agent.update_target()
                q_losses.append(loss)

            state = new_state
            episode_reward += reward
            if done:
                break
        ep_rew, ep_len = test_agent(agent, env_id)
        temp = temp * temp_factor

        rewards.append(episode_reward)
        test_rewards.append(ep_rew)
