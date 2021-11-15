#!/usr/bin/env python
"""
   isort:skip_file
"""
import sys
import os
import platform
from pathlib import Path
from concurrent import futures
import urllib.parse
from joblib import Parallel, delayed, parallel_backend, cpu_count
from sacred import Experiment
import pandas as pd

from sacred.observers import MongoObserver
import numpy as np

# SETTINGS.CAPTURE_MODE = "fd"
# go to parent dir
# sys.path.append("/workspace/src")
if platform.system() == "Windows":
    sys.path.append("C:\\Users\\pontesmo\\Documents\\GitHub\\ddpg_sacred\\src")
    # IP to connect locally. Use "172.25.70.141" if connecting from server
    host = "127.0.0.1"
else:
    from sacred import SETTINGS

    SETTINGS["CAPTURE_MODE"] = "sys"
    # IP to connect locally. Use "172.25.70.141" if connecting from server
    host = "172.25.70.141"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""


from dqn_sacred.runners import main_runner

ex = Experiment("DQN")
# all_ex = Experiment("MADDPG l2t Full")
# My local DB:
# host = "127.0.0.1"
# # username = urllib.parse.quote_plus("mongo_express_user")
# # password = urllib.parse.quote_plus("mongo_express_pw")
# username = urllib.parse.quote_plus("mongo_user")
# password = urllib.parse.quote_plus("mongo_password")
# mongo_db_url = f"mongodb://{username}:{password}@{host}:27017"

username = urllib.parse.quote_plus("pontesmo")
password = urllib.parse.quote_plus("9jyWZ8B3udrfsK3Z")
mongo_db_url = f"mongodb://{username}:{password}@{host}:3308/l2t"
db_name = "l2t"

# all_ex.observers.append(MongoObserver(url=mongo_db_url, db_name=db_name))
ex.observers.append(MongoObserver(url=mongo_db_url, db_name=db_name))


@ex.config
def config():
    seed = 101  # seed
    device = "cpu"
    n_episodes = 1000
    batch_size = 256
    alpha = 0.01
    gamma = 0.9
    epsilon = 0.05


@ex.automain
def run_experiment(n_episodes, device, seed, batch_size, alpha, gamma, epsilon, _run):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env_id = "CartPole-v1"
    # env_id = "LunarLander-v2"
    reward_thr = abs(gym.spec(env_id).reward_threshold)
    env = gym.make(env_id)
    env.seed(seed)
    agent = DQN_Agent(env, alpha=alpha, gamma=gamma)
    temp = epsilon

    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        episode_reward = 0
        for step in range(200):
            action = agent.get_action(state, eps=temp)
            new_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > batch_size:
                loss = agent.update(batch_size)
                _run.log_scalar("loss", loss)
                agent.update_target()

            state = new_state
            episode_reward += reward
            if done:
                break
        test_reward, test_len = test_agent(agent, env_id)

        _run.log_scalar("train_reward", episode_reward)
        _run.log_scalar("test_reward", test_reward)
