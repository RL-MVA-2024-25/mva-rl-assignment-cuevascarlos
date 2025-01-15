import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from copy import deepcopy
from tqdm import tqdm
#import wandb
import matplotlib.pyplot as plt
import os

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return [torch.tensor(np.array(x), dtype=torch.float32, device=self.device) for x in zip(*batch)]

    def __len__(self):
        return len(self.data)

def greedy_action(network, state):
   # Ensure the device is set to match the network's parameters
    device = next(network.parameters()).device
    # Convert the state to a torch tensor and add a batch dimension
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        Q = network(state_tensor)
        return torch.argmax(Q).item()

class ProjectAgent:
    def __init__(self, config= {}, input_dim=6, action_dim=4):
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.nb_actions = action_dim
        self.gamma = config.get('gamma', 0.95)
        self.batch_size = config.get('batch_size', 256)
        buffer_size = config.get('buffer_size', int(1e5))
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.epsilon_max = config.get('epsilon_max', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_stop = config.get('epsilon_decay_period', 1000)
        self.epsilon_delay = config.get('epsilon_delay_decay', 50)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.epsilon = self.epsilon_max

        self.define_model(input_dim, action_dim)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config.get('criterion', torch.nn.MSELoss())
        lr = config.get('learning_rate', 0.001)
        self.optimizer = config.get('optimizer', torch.optim.Adam(self.model.parameters(), lr=lr))
        self.nb_gradient_steps = config.get('gradient_steps', 1)
        self.update_target_strategy = config.get('update_target_strategy', 'replace')
        self.update_target_freq = config.get('update_target_freq', 20)
        self.update_target_tau = config.get('update_target_tau', 0.005)
        self.target_model.load_state_dict(self.model.state_dict())


    def fill_buffer(self, env, num_steps):
        state, _ = env.reset()
        for _ in tqdm(range(num_steps), desc="Filling buffer"):
            action = env.action_space.sample()
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if done or trunc:
                state, _ = env.reset()
                print(f"Done: {done}, Trunc: {trunc}")
            else:
                state = next_state

    def act(self, observation, use_random=False):
        if use_random and random.random() < self.epsilon:
            return random.randint(0, self.nb_actions - 1)
        return greedy_action(self.model, observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path="./DQN_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        state, _ = env.reset()
        best_reward = float("-inf")
        last_update_best_reward = float("-inf")
        step = 0
        flag_env = "NO_RANDOM"
        for episode in range(max_episode):
            episode_cum_reward = 0
            state, _ = env.reset()
            with tqdm(total=200, desc=f"Episode {episode + 1} - {flag_env}", unit="step") as pbar:
                while True:
                    if step > self.epsilon_delay:
                        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

                    action = self.act(state, use_random=True)

                    next_state, reward, done, trunc, _ = env.step(action)
                    self.memory.append(state, action, reward, next_state, done)
                    episode_cum_reward += reward

                    for _ in range(self.nb_gradient_steps):
                        self.gradient_step()

                    if self.update_target_strategy == 'replace':
                        if step % self.update_target_freq == 0 and not episode_cum_reward*10 < last_update_best_reward: 
                            last_update_best_reward = episode_cum_reward
                            self.target_model.load_state_dict(self.model.state_dict())
                    elif self.update_target_strategy == 'ema':
                        target_state_dict = self.target_model.state_dict()
                        model_state_dict = self.model.state_dict()
                        for key in model_state_dict:
                            target_state_dict[key] = self.update_target_tau * model_state_dict[key] + (1 - self.update_target_tau) * target_state_dict[key]
                        self.target_model.load_state_dict(target_state_dict)

                    step += 1
                    pbar.update(1)
                    if done or trunc:
                        episode_return.append(episode_cum_reward)
                        #wandb.log({"Episode": episode + 1, "Return": episode_cum_reward, "Epsilon": self.epsilon})
                        if episode_cum_reward > best_reward:
                            best_reward = episode_cum_reward
                            self.save(f"./dqn_best.pth")
                        if episode % 50 == 0 and episode > 0:
                            if random.random() < 1.0:
                                env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
                                flag_env = "RANDOM"
                            else:
                                env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
                                flag_env = "NO_RANDOM"
                        elif episode % 10 == 0:
                            env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
                            flag_env = "NO_RANDOM"
                        break
                    else:
                        state = next_state
                    
            print(f"Episode {episode + 1} - Reward: {"{:.3e}".format(episode_cum_reward)} - Epsilon: {self.epsilon:.4f}")
            # Save model every 100 episodes
            if episode_cum_reward > 1e10:
                self.save(f"./models_dqn/dqn_{episode}_high.pth")
            
        return episode_return

    def define_model(self, input_dim, output_dim):
        self.model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256,output_dim)
        ).to(self.device)

if __name__ == "__main__":
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    config = {
        'gamma': 0.97,
        'batch_size': 800,
        'buffer_size': int(1e5),
        'epsilon_max': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay_period': 1000,
        'epsilon_delay_decay': 500,
        'learning_rate': 1e-3,
        'gradient_steps': 4,
        'update_target_strategy': 'replace',
        'update_target_freq': 400,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss(),
    }
    #wandb.init(project="FQI-HIV", name="DQN-1")
    agent = ProjectAgent(config=config, input_dim=6, action_dim=4)
    #agent.fill_buffer(env, num_steps=100000)  # Pre-fill buffer
    episode_return = agent.train(env, max_episode=1000)
    agent.save(f"./models_dqn/dqn_final.pth")
