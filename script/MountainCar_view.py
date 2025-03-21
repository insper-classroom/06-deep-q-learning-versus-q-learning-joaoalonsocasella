import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

####################################################################################################
data_dir = Path("results")
qtable_dir = Path("qtables")


####################################################################################################

def load_dqn_rewards():
    dql_rewards = []
    for i in range(5):
        file_path = data_dir / f"mountaincar_DeepQLearning_rewards_run_{i}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["Run"] = i
            df["Reward_MA"] = df["Reward"].rolling(window=125, min_periods=1).mean() 
            dql_rewards.append(df)
    return pd.concat(dql_rewards, ignore_index=True)

####################################################################################################

def load_q_learning_rewards():
    q_rewards = []
    for i in range(10):
        file_path = qtable_dir / f"Q_table_exec_{i}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["Run"] = i
            q_rewards.append(df)
    return pd.concat(q_rewards, ignore_index=True)

####################################################################################################

df_dqn = load_dqn_rewards()
df_q = load_q_learning_rewards()

####################################################################################################

def process_rewards(df):
    agg = df.groupby("Episode")["Reward"].agg(["mean", "std"]).reset_index()
    agg["Reward_Mean_Smooth"] = agg["mean"].rolling(window=75, min_periods=1).mean()
    agg["Reward_SD_Smooth"]   = agg["std"].rolling(window=75, min_periods=1).mean()
    agg["Upper"] = agg["Reward_Mean_Smooth"] + agg["Reward_SD_Smooth"]
    agg["Lower"] = agg["Reward_Mean_Smooth"] - agg["Reward_SD_Smooth"]
    return agg

agg_dqn = process_rewards(df_dqn)



agg_q = process_rewards(df_q)
####################################################################################################

# RAW

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

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_dqn, x="Episode", y="Reward_MA", hue="Run", palette="tab10", linewidth=2)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho do Deep Q-Learning - Média Móvel (125 episódios)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média (Móvel)")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_dqn_ma125.png", dpi=300)
plt.show()

####################################################################################################

# Desempenho médio

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
plt.savefig("images/comparacao_dqn_media.png", dpi=300)
plt.show()

####################################################################################################

# Gráfico comparativo de desempenho médio entre DQL e QL

plt.figure(figsize=(14, 6))
plt.plot(agg_dqn["Episode"], agg_dqn["Reward_Mean_Smooth"], label="Deep Q-Learning (Média)", color="darkred", linewidth=2)
plt.fill_between(agg_dqn["Episode"], agg_dqn["Lower"], agg_dqn["Upper"], color="darkred", alpha=0.2, label="±1 Desvio Padrão (DQL)")

plt.plot(agg_q["Episode"], agg_q["Reward_Mean_Smooth"], label="Q-Learning (Média)", color="navy", linewidth=2)
plt.fill_between(agg_q["Episode"], agg_q["Lower"], agg_q["Upper"], color="navy", alpha=0.2, label="±1 Desvio Padrão (Q-Learning)")

plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Comparação de Desempenho: Deep Q-Learning vs Q-Learning (MountainCar)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média Suavizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
Path("images").mkdir(exist_ok=True)
plt.savefig("images/comparacao_dql_vs_qlearning.png", dpi=300)
plt.show()