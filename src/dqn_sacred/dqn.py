import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpg_sacred.utils import Memory
from torch_optimizer import RAdam, Ranger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_Net(nn.Module):
    def __init__(self, n_input, n_output, hidden_size=64):
        super(Q_Net, self).__init__()
        self.state_dim = n_input
        self.action_dim = n_output
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DQN_Agent(object):
    def __init__(self, env, alpha=0.01, gamma=0.9, max_memory=50000):
        self.n_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.alpha = alpha
        self.gamma = gamma

        self.memory = Memory(max_memory)
        self.q_net = Q_Net(self.state_dim, self.n_actions).to(DEVICE)
        self.q_net_target = Q_Net(self.state_dim, self.n_actions).to(DEVICE)

        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = Ranger(self.q_net.parameters())
        self.criterion = nn.SmoothL1Loss()

    def get_action(self, state, eps=0.1):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(DEVICE)
            q_values = self.q_net(state).numpy()
        if np.random.rand() >= eps:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float).to(DEVICE)
        actions = torch.tensor(actions).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float).to(DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.float).to(DEVICE)

        with torch.no_grad():
            next_q = self.q_net_target(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * (1 - dones) * next_q
        q_value = self.q_net(states)
        q_value = q_value.gather(1, actions.unsqueeze(-1))
        loss = self.criterion(q_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().to("cpu").numpy()

    def update_target(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())
