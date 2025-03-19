import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gc

class DeepQLearning:
    def __init__(self,
                 env, gamma, epsilon, epsilon_min,
                 epsilon_dec, episodes, batch_size,
                 memory, model, max_steps, device="cuda"):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model.to(device)
        self.max_steps = max_steps
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()
    
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
    
    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            states        = torch.tensor(np.array([i[0] for i in batch]), dtype=torch.float32, device=self.device)
            actions       = torch.tensor(np.array([i[1] for i in batch]), dtype=torch.long, device=self.device)
            rewards       = torch.tensor(np.array([i[2] for i in batch]), dtype=torch.float32, device=self.device)
            next_states   = torch.tensor(np.array([i[3] for i in batch]), dtype=torch.float32, device=self.device)
            terminals     = torch.tensor(np.array([i[4] for i in batch]), dtype=torch.float32, device=self.device)

            # Obter os valores Q estimados pela rede para os estados atuais
            
            print(f"Shape de q_values antes do gather: {self.model(states).shape}")
            print(f"Shape de actions antes do view: {actions.shape}")
            print(f"Shape de actions após o view: {actions.view(-1, 1).shape}")
            q_values = self.model(states).gather(1, actions.view(-1, 1)).squeeze(1)

            # Calcular os valores Q esperados para os próximos estados (somente para estados não terminais)
            with torch.no_grad():
                next_max_q_values = self.model(next_states).max(1)[0]

            # Atualizar os valores-alvo usando a equação de Bellman
            targets = rewards + self.gamma * next_max_q_values * (1 - terminals.bool().float())

            # Calcular a perda entre os valores Q previstos e os valores-alvo
            loss = self.criterion(q_values, targets.detach())

            # Backpropagation para atualizar os pesos da rede
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decaimento do epsilon para reduzir exploração com o tempo
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec


    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            state, _ = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            steps = 0
            done = False
            
            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or truncated or (steps > self.max_steps):
                    done = True          
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                
                if done:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            
            rewards.append(score)
            gc.collect()
        
        return rewards