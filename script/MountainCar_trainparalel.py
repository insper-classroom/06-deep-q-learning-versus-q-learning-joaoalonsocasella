import torch
torch.cuda.set_per_process_memory_fraction(0.2, device=0)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import gc
import sys
import csv
import multiprocessing
from collections import deque
from pathlib import Path

path = r"C:/RL/06-deep-q-learning-versus-q-learning-joaoalonsocasella"
sys.path.append(path)
from FUNCTIONS.DeepQLearning import DeepQLearning as f_DQN
import importlib
import FUNCTIONS.DeepQLearning
importlib.reload(FUNCTIONS.DeepQLearning)

# Parâmetros fixos
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.995
episodes = 10000
batch_size = 32
learning_rate = 0.001
max_steps = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo da rede
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)

# Função de treino individual por processo
def run_training(process_id):
    torch.cuda.empty_cache()
    env = gym.make('MountainCar-v0', max_episode_steps=1000)
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    memory = deque(maxlen=10_000)

    agent = f_DQN(env, gamma, epsilon, epsilon_min, epsilon_dec,
                  episodes, batch_size, memory, model, max_steps, device=device)
    rewards = agent.train()

    # Salvar CSV
    Path("results").mkdir(exist_ok=True)
    with open(f'results/mountaincar_DeepQLearning_rewards_run_{process_id}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])
        for episode, reward in enumerate(rewards):
            print(f"[Rodagem {process_id}] Episódio {episode+1}: Score = {reward}", flush=True)
            writer.writerow([episode, reward])

    # Salvar modelo
    Path("data").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f'data/model_mountaincar_run_{process_id}.pth')

# Execução paralela
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Necessário no Windows
    num_runs = 5
    processes = []
    for i in range(num_runs):
        p = multiprocessing.Process(target=run_training, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

# PARA RODAR, JOGAR NO TERMINAL: python -u  .\script\MountainCar_trainparalel.py


