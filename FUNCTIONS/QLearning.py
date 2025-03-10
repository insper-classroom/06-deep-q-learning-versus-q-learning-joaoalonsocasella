import gymnasium as gym
import numpy as np
import pandas as pd
import os


class QLearning:
    def __init__(self, env, gamma, alpha, epsilon, epsilon_min, epsilon_dec, episodios):
        """
        Classe para implementar o algoritmo Q-Learning.
        
        Parâmetros:
        - env: ambiente Gymnasium
        - gamma: fator de desconto
        - alpha: taxa de aprendizado
        - epsilon: taxa de exploração inicial
        - epsilon_min: taxa mínima de exploração
        - epsilon_dec: fator de decaimento do epsilon
        - episodios: número de episódios de treinamento
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodios = episodios
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.Q_table = np.zeros((self.num_states, self.num_actions))



    def select_action(self, state):
        """ Escolhe uma ação baseada na política ε-greedy. """
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state, :])


    def train(self):
        """
        Treina o agente usando Q-Learning.
        
        Retorna:
        - log_rewards: lista de recompensas acumuladas por episódio
        - Q_table: tabela de valores Q aprendida
        """
        log_rewards = []
        for episode in range(self.episodios):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done and total_reward > -2000:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                best_next_action = np.max(self.Q_table[next_state, :])
                self.Q_table[state, action] += self.alpha * (reward + self.gamma * best_next_action - self.Q_table[state, action])
                total_reward += reward
                state = next_state
            log_rewards.append(total_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)
        self.env.close()
        return log_rewards, self.Q_table

    def save_q_table(self, file_path="Q_tables/QLearning/Q_table.csv"):
        """ Salva a Q-table treinada em um arquivo CSV. """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pd.DataFrame(self.Q_table).to_csv(file_path, index=False)

    def load_q_table(self, file_path="Q_tables/QLearning/Q_table.csv"):
        """ Carrega uma Q-table previamente treinada de um arquivo CSV. """
        if os.path.exists(file_path):
            self.Q_table = pd.read_csv(file_path).values
            print(f"Q-table carregada de {file_path}.")
        else:
            print(f"Arquivo {file_path} não encontrado. Iniciando com Q-table zerada.")
            
    def test(self, num_episodes=100):
        """
        Testa o agente treinado sem exploração (epsilon = 0).
        
        Retorna:
        - lista com as recompensas acumuladas por episódio.
        """
        test_rewards = []
        epsilon_original = self.epsilon
        self.epsilon = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.Q_table[state, :])
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
            test_rewards.append(total_reward)
        self.epsilon = epsilon_original  # Restaura o epsilon original
        return test_rewards
############################################
# Função para executar múltiplos treinamentos e salvar as estatísticas
def train_multiple_runs(n_runs=5, env_name='Taxi-v3'):
    """
    Executa múltiplos treinamentos de Q-Learning e salva os resultados.

    Parâmetros:
    - n_runs: número de execuções do treinamento
    - env_name: nome do ambiente Gymnasium
    """
    env = gym.make(env_name)
    df_list = []
    for i in range(n_runs):
        agent = QLearning(env)
        rewards, Q_table = agent.train()
        agent.save_q_table(f"Q_tables/QLearning/Q_table_Q_Learning_run_{i}.csv")
        df_temp = pd.DataFrame({
            "Episódio": np.arange(len(rewards)),
            "Recompensa": rewards,
            "Execução": f"Q-Learning {i}"
        })
        df_list.append(df_temp)
    df_results = pd.concat(df_list)
    df_results.to_csv("results/q_learning_training_results.csv", index=False)
    print("Treinamento concluído e resultados salvos!")