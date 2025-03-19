import numpy as np
import gymnasium as gym

class QLearning:
    def __init__(self, env, gamma, alpha, epsilon=1.0, epsilon_min=0.05, epsilon_dec=0.999, episodes=20000):
        self.env = env
        self.gamma = gamma  # Fator de desconto
        self.alpha = alpha  # Taxa de aprendizado
        self.epsilon = epsilon  # Probabilidade de exploração
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        
        # 🔹 Discretizando o espaço contínuo
        self.num_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.num_states = np.round(self.num_states, 0).astype(int) + 1
        self.Q = np.zeros([self.num_states[0], self.num_states[1], env.action_space.n])

    def transform_state(self, state):
        """Converte o estado contínuo em índices discretos."""
        state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state_adj, 0).astype(int)

    def select_action(self, state_adj):
        """Política epsilon-greedy."""
        if np.random.random() < 1 - self.epsilon:
            return np.argmax(self.Q[state_adj[0], state_adj[1]])
        return np.random.randint(0, self.env.action_space.n)

    def update_q_table(self, state_adj, action, reward, next_state_adj, done):
        best_next_action = np.argmax(self.Q[next_state_adj[0], next_state_adj[1]])
        td_target = reward + self.gamma * self.Q[next_state_adj[0], next_state_adj[1], best_next_action] * (1 - done)
        self.Q[state_adj[0], state_adj[1], action] += self.alpha * (td_target - self.Q[state_adj[0], state_adj[1], action])

    def train(self):
        """Treina o agente e retorna as recompensas por episódio."""
        rewards = []
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            state_adj = self.transform_state(state)
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state_adj)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state_adj = self.transform_state(next_state)
                
                self.update_q_table(state_adj, action, reward, next_state_adj, done)
                state_adj = next_state_adj
                total_reward += reward

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec
            
            rewards.append(total_reward)
            print(f'Episódio {episode+1}/{self.episodes}, Recompensa: {total_reward}')
        
        return rewards
