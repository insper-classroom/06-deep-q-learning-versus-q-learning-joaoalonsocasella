import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

####################################################################################################
# Diretórios
data_dir = Path("results")
images_dir = Path("images")
images_dir.mkdir(exist_ok=True) 
####################################################################################################

def load_dqn_rewards():
    dqn_rewards = []
    for i in range(5):  # Consideramos 5 rodadas do DQN
        file_path = data_dir / f"mountaincar_DeepQLearning_rewards_run_{i}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["Run"] = i
            df["Reward_MA"] = df["Reward"].rolling(window=125, min_periods=1).mean()  # Média móvel de 125 episódios
            dqn_rewards.append(df)
    return pd.concat(dqn_rewards, ignore_index=True) if dqn_rewards else None


####################################################################################################

def load_q_learning_rewards():
    q_rewards = []
    for i in range(5):  # Consideramos 5 rodadas do Q-Learning
        file_path = data_dir / f"mountaincar_QLearning_rewards_run_{i}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["Run"] = i
            df["Reward_MA"] = df["Reward"].rolling(window=125, min_periods=1).mean()
            q_rewards.append(df)
    return pd.concat(q_rewards, ignore_index=True) if q_rewards else None
####################################################################################################

df_dqn = load_dqn_rewards()
df_q = load_q_learning_rewards()

if df_dqn is None or df_q is None:
    raise FileNotFoundError("Erro: Arquivos CSV não encontrados para DQN ou Q-Learning!")

####################################################################################################

def process_rewards(df):
    agg = df.groupby("Episode")["Reward"].agg(["mean", "std"]).reset_index()
    agg["Reward_Mean_Smooth"] = agg["mean"].rolling(window=75, min_periods=1).mean()
    agg["Reward_SD_Smooth"] = agg["std"].rolling(window=75, min_periods=1).mean()
    agg["Upper"] = agg["Reward_Mean_Smooth"] + agg["Reward_SD_Smooth"]
    agg["Lower"] = agg["Reward_Mean_Smooth"] - agg["Reward_SD_Smooth"]
    return agg

agg_dqn = process_rewards(df_dqn)
agg_q = process_rewards(df_q)
####################################################################################################

# RAW

# Deep-Q-Learning

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_dqn, x="Episode", y="Reward", hue="Run", palette="tab10", linewidth=1.8)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa')
plt.title("Desempenho do Deep Q-Learning no MountainCar (5 rodagens)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/DQN_raw.png", dpi=300)
plt.show()

# Q-Learning

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_q, x="Episode", y="Reward", hue="Run", palette="tab10", linewidth=1.8)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa')
plt.title("Desempenho do Q-Learning no MountainCar (5 rodagens)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/QL_raw.png", dpi=300)
plt.show()

####################################################################################################
# MA 125

# Deep-Q-Learning

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_dqn, x="Episode", y="Reward_MA", hue="Run", palette="tab10", linewidth=2)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho do Deep Q-Learning - Média Móvel (125 episódios)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média (Móvel)")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_DQN_ma125.png", dpi=300)
plt.show()


# Q-Learning

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_q, x="Episode", y="Reward_MA", hue="Run", palette="tab10", linewidth=2)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho do Q-Learning - Média Móvel (125 episódios)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média (Móvel)")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_QL_ma125.png", dpi=300)
plt.show()


####################################################################################################

# Desempenho médio

# Deep-Q-Learning

agg = (
    df_dqn.groupby("Episode")["Reward"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "Reward_Mean", "std": "Reward_SD"})
)
agg["Reward_Mean_Smooth"] = agg["Reward_Mean"].rolling(window=75, min_periods=1).mean()
agg["Reward_SD_Smooth"]   = agg["Reward_SD"].rolling(window=75, min_periods=1).mean()
agg["Upper"] = agg["Reward_Mean_Smooth"] + agg["Reward_SD_Smooth"]
agg["Lower"] = agg["Reward_Mean_Smooth"] - agg["Reward_SD_Smooth"]

plt.figure(figsize=(14, 6))
plt.plot(agg["Episode"], agg["Reward_Mean_Smooth"], label="Recompensa Média (5 rodagens)", color="darkred", linewidth=2)
plt.fill_between(agg["Episode"], agg["Lower"], agg["Upper"], color="darkred", alpha=0.2, label="±1 Desvio Padrão")
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho Médio do Deep Q-Learning com Variância (MountainCar)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média Suavizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_DQN_media.png", dpi=300)
plt.show()



# Q-Learning

agg = (
    df_q.groupby("Episode")["Reward"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "Reward_Mean", "std": "Reward_SD"})
)
agg["Reward_Mean_Smooth"] = agg["Reward_Mean"].rolling(window=75, min_periods=1).mean()
agg["Reward_SD_Smooth"]   = agg["Reward_SD"].rolling(window=75, min_periods=1).mean()
agg["Upper"] = agg["Reward_Mean_Smooth"] + agg["Reward_SD_Smooth"]
agg["Lower"] = agg["Reward_Mean_Smooth"] - agg["Reward_SD_Smooth"]

plt.figure(figsize=(14, 6))
plt.plot(agg["Episode"], agg["Reward_Mean_Smooth"], label="Recompensa Média (5 rodagens)", color="darkblue", linewidth=2)
plt.fill_between(agg["Episode"], agg["Lower"], agg["Upper"], color="darkblue", alpha=0.2, label="±1 Desvio Padrão")
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho Médio do Deep Q-Learning com Variância (MountainCar)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média Suavizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_QL_media.png", dpi=300)
plt.show()

####################################################################################################

# Gráfico comparativo de desempenho médio entre DQL e QL

plt.figure(figsize=(14, 6))
plt.plot(agg_dqn["Episode"], agg_dqn["Reward_Mean_Smooth"], label="Deep Q-Learning (Média)", color="darkred", linewidth=2)
plt.fill_between(agg_dqn["Episode"], agg_dqn["Lower"], agg_dqn["Upper"], color="darkred", alpha=0.2, label="±1 Desvio Padrão (DQN)")
plt.plot(agg_q["Episode"], agg_q["Reward_Mean_Smooth"], label="Q-Learning (Média)", color="navy", linewidth=2)
plt.fill_between(agg_q["Episode"], agg_q["Lower"], agg_q["Upper"], color="navy", alpha=0.2, label="±1 Desvio Padrão (Q-Learning)")
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Comparação de Desempenho: Deep Q-Learning vs Q-Learning (MountainCar)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média Suavizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(images_dir / "comparacao_DQN_vs_QL.png", dpi=300)
plt.show()
