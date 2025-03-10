import gymnasium as gym
import numpy as np
import keras
from keras import models

# Carregar o ambiente em modo de renderização para visualizar a simulação
env = gym.make('MountainCar-v0', render_mode='human').env

# Reiniciar o ambiente e carregar o modelo treinado
(state, _) = env.reset()
model = models.load_model('data/model_mountaincar.keras')

# Inicializar variáveis
done = False
truncated = False
total_rewards = 0
steps = 0
max_steps = 500

# Loop de interação com o ambiente até atingir a condição de parada
while (not done) and (not truncated) and (steps < max_steps):
    # Predição da melhor ação com a rede neural treinada
    Q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(Q_values[0])

    # Executar a ação no ambiente
    state, reward, done, truncated, _ = env.step(action)
    total_rewards += reward

    # Renderizar o ambiente
    env.render()
    steps += 1

# Exibir o resultado final
print(f'Score final = {total_rewards}')
input('Pressione qualquer tecla para sair...')
