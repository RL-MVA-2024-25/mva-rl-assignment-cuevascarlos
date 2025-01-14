import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
# import wandb
import pickle
import os
from tqdm import tqdm
import random
import joblib
from huggingface_hub import hf_hub_download

class ProjectAgent:
    def __init__(self, env=None, gamma=0.99, epsilon=1.0, epsilon_min=0.05, decay_rate=0.995, epsilon_delay_decay=50, max_buffer_size=300000):  
        self.models = [RandomForestRegressor(n_estimators=50, random_state=42) for _ in range(4)]
        self.state_dim = 6
        self.action_dim = 4
        self.scalers = [StandardScaler() for _ in range(self.action_dim)]
        self.actions = list(range(env.action_space.n)) if env else [0, 1, 2, 3]
        self.initialized = [False for _ in range(4)]
        self.gamma = gamma

        self.max_buffer_size = max_buffer_size
        self.buffer = []

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.epsilon_delay = epsilon_delay_decay

    def act(self, observation, use_random=False):
        #if (use_random or not self.initialized) and random.random() < self.epsilon:
        #    return random.choice(self.actions)
        if use_random and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        q_values = []
        for a in self.actions:
            try:
                obs_scaled = self.scalers[a].transform(np.hstack([observation]).reshape(1, -1))
                q_value = self.models[a].predict(obs_scaled)[0]
            except:
                q_value = float('-inf')
            q_values.append(q_value)
        
        return int(np.argmax(q_values))
    
    def set_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def collect_transitions(self, steps, env, use_random=True):
        transitions = []
        state, _ = env.reset()
        
        for _ in tqdm(range(steps)):
            action = self.act(state, use_random)
            next_state, reward, done, trunc, _ = env.step(action)

            transitions.append((state, action, reward, next_state, done))

            if done or trunc:
                state, _ = env.reset()
            else:
                state = next_state
                
        return transitions

    def _prepare_fqi_dataset(self, transitions):
        states = np.vstack([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.vstack([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])

        action_datasets = [[] for _ in range(self.action_dim)]
        action_targets = [[] for _ in range(self.action_dim)]

        next_q_values = np.zeros((len(states), self.action_dim))
        if self.initialized[0]:
            for a in range(self.action_dim):
                next_states_scaled = self.scalers[a].transform(next_states)
                next_q_values[:, a] = self.models[a].predict(next_states_scaled)

        max_next_q = np.max(next_q_values, axis=1)

        for i in range(len(states)):
            action = int(actions[i])
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * max_next_q[i]

            action_datasets[action].append(states[i])
            action_targets[action].append(target)

        return action_datasets, action_targets

    def evaluate(self, env, num_episodes=5):
        total_reward = 0
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = env.reset()
            done = False
            trunc = False
            while not done and not trunc:
                action = self.act(state, use_random=False)
                next_state, reward, done, trunc, _ = env.step(action)
                total_reward += reward
                state = next_state
        return total_reward / num_episodes

    def save(self, name, path = None):
        if path is None:
            path = f"./{name}.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'models': self.models,
            'scalers': self.scalers
        }
        joblib.dump(save_dict, path)

    def load(self, name="RF_model_rounded", path = None):  
        model_path = hf_hub_download(repo_id="cuevascarlos/HIV-RL-RandomForest-policy", filename = "RF_model.pkl")

        if path is None:
            path = f"./{name}.pkl"
        save_dict = joblib.load(model_path)
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']
        self.initialized = [True for _ in range(4)]

    def fit_models(self, action_datasets, action_targets):
        for a in range(self.action_dim):
            print(f"Training model {a} with {len(action_datasets[a])} samples")
            if len(action_datasets[a]) > 0:
                X = np.array(action_datasets[a])
                y = np.array(action_targets[a])

                X_scaled = self.scalers[a].fit_transform(X)
                self.models[a].fit(X_scaled, y)
                self.initialized[a] = True

    def bootstrap_creation(self):
        bootstrap_idx = np.random.choice(len(self.buffer), len(self.buffer), replace=True)
        bootstrap_transitions = [self.buffer[i] for i in bootstrap_idx]
        action_datasets, action_targets = self._prepare_fqi_dataset(bootstrap_transitions)
        return action_datasets, action_targets

    def train(self, env, num_epochs=100, episodes_per_epoch=100, initialize_buffer=False):

        if initialize_buffer:
            self.buffer = self.collect_transitions(self.max_buffer_size, env, use_random=True)
            # Fit models
            action_datasets, action_targets = self.bootstrap_creation()
            self.fit_models(action_datasets, action_targets)
            agent.set_epsilon()
            
        best_eval_reward = float('-inf')

        for epoch in range(num_epochs):
            epoch_rewards = []

            for episode in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                episode_transitions = self.collect_transitions(200, env, use_random=False)
                episode_reward = sum(t[2] for t in episode_transitions)
                self.buffer.extend(episode_transitions)
                if len(self.buffer) > self.max_buffer_size:
                    self.buffer = self.buffer[-self.max_buffer_size:]
                epoch_rewards.append(episode_reward)
                agent.set_epsilon()

            action_datasets, action_targets = self.bootstrap_creation()
            self.fit_models(action_datasets, action_targets)

            eval_reward = self.evaluate(env, num_episodes=10)
            avg_reward = np.mean(epoch_rewards)

            print(f"Epoch {epoch + 1}  | Summary: Avg Epoch Reward: {avg_reward:.2e}, Eval Reward: {eval_reward:.2e}, Epsilon: {self.epsilon:.2e}")
            # wandb.log({"epoch": epoch, "avg_reward": avg_reward, "eval_reward": eval_reward})

            # self.save(f"model_{epoch+1}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"Best eval reward: {epoch+1} - Value: {best_eval_reward}")
                self.save(f"best_eval_model")
        
        self.save("final_model")

if __name__ == "__main__":
    # wandb.init(project="FQI-HIV", name="combined_training")
    env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
    agent = ProjectAgent()
    agent.train(env, num_epochs=100, episodes_per_epoch=100, initialize_buffer=True)
    # wandb.finish()