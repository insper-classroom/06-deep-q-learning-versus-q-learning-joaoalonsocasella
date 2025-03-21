import torch
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
path = r"C:/RL/06-deep-q-learning-versus-q-learning-joaoalonsocasella"
# path = r"C:/Users/João Alonso Casella/OneDrive/Joao/Insper/Eletivas/Reinforcement_Learning/06-deep-q-learning-versus-q-learning-joaoalonsocasella-1"
sys.path.append(path)
from FUNCTIONS.DeepQLearning import DeepQLearning as f_DQN
####################
import importlib
import FUNCTIONS.DeepQLearning
importlib.reload(FUNCTIONS.DeepQLearning)
####################
# Ambiente
env = gym.make('MountainCar-v0', max_episode_steps=1000)
np.random.seed(0)
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

####################
# Parâmetros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.995
episodes = 10000
batch_size = 64
learning_rate = 0.001
memory = deque(maxlen=10_000)
max_steps = 500
N = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
# Definição do modelo
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

dqn_model  = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(dqn_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"Modelo está na GPU? {next(dqn_model.parameters()).is_cuda}")  # Deve imprimir True
 # Deve imprimir True



####################
# Treinamento
dqn_rewards = []

for i in range(N):
    print(f"Treinamento {i+1}/{N} - Deep Q-Learning")
    dqn_agent  = f_DQN(env, gamma, epsilon,
                epsilon_min, epsilon_dec,
                episodes, batch_size, memory,
                dqn_model , max_steps)
    rewards_d = dqn_agent.train()
    dqn_rewards.append(rewards_d)

    # Salvar as recompensas do Deep Q-Learning em CSV
    with open(f'results/mountaincar_DeepQLearning_rewards_run_{i}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])
        for episode, reward in enumerate(rewards_d):
            writer.writerow([episode, reward])

# Salvar modelo treinado da rede neural
torch.save(dqn_model.state_dict(), 'data/model_mountaincar.pth')

# Plot comparativo
# import matplotlib.pyplot as plt
# plt.plot(np.mean(dqn_rewards, axis=0), label='Deep Q-Learning')
# plt.axhline(y=-110, color='r', linestyle='--', label='Meta de Recompensa')
# plt.xlabel('Episodes')
# plt.ylabel('Recompensa Acumulada')
# plt.legend()
# plt.title('Desempenho do Deep Q-Learning - MountainCar')
# plt.savefig("results/comparacao_dqn.jpg")
# plt.show()

##############################################################################################################

# Testar o modelo treinado
# env = gym.make('MountainCar-v0', render_mode='human').env
# (state, _) = env.reset()
# dqn_model.load_state_dict(torch.load('data/model_mountaincar.pth'))
# dqn_model.eval()

# done = False
# truncated = False
# rewards = 0
# steps = 0
# max_steps = 500

# while (not done) and (not truncated) and (steps < max_steps):
#     state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#     with torch.no_grad():
#         Q_values = dqn_model(state_tensor)
#     action = torch.argmax(Q_values).item()
#     state, reward, done, truncated, info = env.step(action)
#     rewards += reward
#     env.render()
#     steps += 1

# print(f'Score = {rewards}')
# input('press a key...')
