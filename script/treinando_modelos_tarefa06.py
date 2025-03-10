import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import random
import keras
import tensorflow as tf
import sys
import csv
from tensorflow import keras
from keras import optimizers
from collections import deque
from keras import models, layers, relu, linear
path = r"C:/Users/João Alonso Casella/OneDrive/Joao/Insper/Eletivas/Reinforcement_Learning/06-deep-q-learning-versus-q-learning-joaoalonsocasella"
if path not in sys.path:"C:/Users/João Alonso Casella/OneDrive/Joao/Insper/Eletivas/Reinforcement_Learning/06-deep-q-learning-versus-q-learning-joaoalonsocasella"
sys.path.append(path)

from FUNCTIONS.DeepQLearning import DeepQLearning
from FUNCTIONS.QLearning import QLearning



####################
# Em aula vimos como foi feito o treinamento usando Deep-Q-Learning
# para o ambiente de Cartpole. Agora, implementa-se o mesmo treinamento
# para o ambiente MountainCar. Para isso, é necessário fazer algumas
# modificações no código original.
####################


####################
# Ambiente
env = gym.make('MountainCar-v0', max_episode_steps=1000)
#env.seed(0)
np.random.seed(0)
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

####################
# Parâmetros
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.99
episodes = 200
batch_size = 64
learning_rate = 0.001
memory = deque(maxlen=10000)
max_steps = 500
N = 5

####################
# Definição do modelo
model = models.Sequential([
    layers.Dense(512, activation=relu, input_dim=env.observation_space.shape[0]),
    layers.Dense(256, activation=relu),
    layers.Dense(env.action_space.n, activation=linear)
])
model.compile(loss='mse', optimizer='adam')

####################

# Treinamento

q_learning_rewards = []
dqn_rewards = []

for i in range(N):
    print(f"Treinamento {i+1}/{N} - Q-Learning")
    QL = QLearning(env, gamma, learning_rate, epsilon, epsilon_min, epsilon_dec, episodes)
    rewards_q, _ = QL.train()
    q_learning_rewards.append(rewards_q)

    print(f"Treinamento {i+1}/{N} - Deep Q-Learning")
    DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps)
    rewards_d = DQN.train()
    dqn_rewards.append(rewards_d)

    # Salvar as recompensas do Deep Q-Learning em CSV
    with open(f'results/mountaincar_DeepQLearning_rewards_run_{i}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])
        for episode, reward in enumerate(rewards_d):
            writer.writerow([episode, reward])

# Salvar modelo treinado da rede neural
model.save('data/model_mountaincar.keras')

# Plot comparativo
plt.plot(np.mean(q_learning_rewards, axis=0), label='Q-Learning')
plt.plot(np.mean(dqn_rewards, axis=0), label='Deep Q-Learning')
plt.axhline(y=-110, color='r', linestyle='--', label='Meta de Recompensa')
plt.xlabel('Episodes')
plt.ylabel('Recompensa Acumulada')
plt.legend()
plt.title('Comparação Q-Learning vs Deep Q-Learning - MountainCar')
plt.savefig("results/comparacao_qlearning_vs_dqn.jpg")
plt.show()

##############################################################################################################

# Testar o modelo treinado
env = gym.make('MountainCar-v0', render_mode='human').env
(state,_) = env.reset()
model = keras.models.load_model('data/model_mountaincar.keras')
done = False
truncated = False
rewards = 0
steps = 0
max_steps = 500

while (not done) and (not truncated) and (steps<max_steps):
    Q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(Q_values[0])
    state, reward, done, truncated, info = env.step(action)
    rewards += reward
    env.render()
    steps += 1

print(f'Score = {rewards}')
input('press a key...')