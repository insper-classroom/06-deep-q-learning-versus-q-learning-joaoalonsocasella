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

env = gym.make('CartPole-v1')
#env.seed(0)
np.random.seed(0)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

model = keras.Sequential()
model.add(keras.layers.Dense(512, activation=relu, input_dim=env.observation_space.shape[0]))
model.add(keras.layers.Dense(256, activation=relu))
model.add(keras.layers.Dense(env.action_space.n, activation=linear))
model.summary()
model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))

gamma = 0.99 
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.99
episodes = 200
batch_size = 64
memory = deque(maxlen=10000) #talvez usar uma memoria mais curta
max_steps = 500

DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps)
rewards = DQN.train()

import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('# Rewards')
plt.title('# Rewards vs Episodes')
plt.savefig("results/cartpole_DeepQLearning.jpg")     
plt.close()

with open('results/cartpole_DeepQLearning_rewards.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    episode=0
    for reward in rewards:
        writer.writerow([episode,reward])
        episode+=1

model.save('data/model_cart_pole.keras')

