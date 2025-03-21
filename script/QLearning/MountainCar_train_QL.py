import gymnasium as gym
import numpy as np
import torch
import multiprocessing
import os
import pandas as pd
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# python -u .\script\QLearning\MountainCar_train_QL.py

# Hiperparâmetros
best_alpha = 0.2
best_gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_dec = 0.997
episodes = 10000
max_steps = 500
n_exec = 5
n_test_episodes = 50

# Diretórios
output_dir = Path("data/qtables")
results_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Discretização do estado
def discretize_state(state, env):
    num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1
    state_adj = (state - env.observation_space.low) * np.array([10, 100])
    return tuple(np.round(state_adj, 0).astype(int))

# Classe do Agente
class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        self.Q = torch.zeros((self.num_states[0], self.num_states[1], env.action_space.n), device=device)

    def select_action(self, state_adj):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        return torch.argmax(self.Q[state_adj[0], state_adj[1]]).item()

    def train(self, process_id):
        rewards = []
        for ep in range(self.episodes):
            state, _ = self.env.reset()
            state_adj = discretize_state(state, self.env)
            total_reward = 0
            steps = 0
            done = False
            while not done and steps < max_steps:
                steps += 1
                action = self.select_action(state_adj)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state_adj = discretize_state(next_state, self.env)
                # Bônus por se aproximar do objetivo
                if next_state[0] >= 0.5:
                    reward += 100
                with torch.no_grad():
                    best_next_q = torch.max(self.Q[next_state_adj[0], next_state_adj[1]])
                    self.Q[state_adj[0], state_adj[1], action] += self.alpha * (
                        reward + self.gamma * best_next_q - self.Q[state_adj[0], state_adj[1], action])
                state_adj = next_state_adj
                total_reward += reward
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
            rewards.append(total_reward)
            if ep % 100 == 0:
                print(f"[Q-Learning] Execução {process_id} - Episódio {ep}/{self.episodes} - Recompensa: {total_reward}")

        return rewards

    def save_q_table(self, filename_pth):
        torch.save(self.Q, filename_pth)

# Função de execução paralela
def run_training(process_id):
    torch.cuda.empty_cache()
    env = gym.make('MountainCar-v0', max_episode_steps=1000)
    agent = QLearningAgent(env, best_alpha, best_gamma, epsilon_start, epsilon_min, epsilon_dec, episodes)
    rewards = agent.train(process_id)

    # Salva Q-table (apenas em .pth)
    qtable_pth = output_dir / f"Q_table_exec_{process_id}.pth"
    agent.save_q_table(qtable_pth)

    # Salva recompensas por episódio em CSV com nome correto
    rewards_df = pd.DataFrame({
        "Episode": np.arange(1, len(rewards)+1),
        "Reward": rewards,
        "Run": process_id
    })
    rewards_csv = results_dir / f"mountaincar_QLearning_rewards_run_{process_id}.csv"
    rewards_df.to_csv(rewards_csv, index=False)
    print(f"Execução {process_id} concluída. Resultados salvos em {rewards_csv}.")

# Execução principal
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    ctx = multiprocessing.get_context("spawn")
    processes = []

    for i in range(n_exec):
        p = ctx.Process(target=run_training, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Treinamento Q-Learning concluído! Recompensas salvas em 'results/'.")
