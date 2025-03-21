import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


data_dir = Path("results")
all_runs = []
for i in range(5):
    file_path = data_dir / f"mountaincar_DeepQLearning_rewards_run_{i}.csv"
    if not file_path.exists():
        print(f"[!] Arquivo não encontrado: {file_path}")
        continue
    df = pd.read_csv(file_path)
    df["Run"] = f"Rodagem {i+1}"
    df["Reward_MA"] = df["Reward"].rolling(window=125, min_periods=1).mean()  # Média móvel de 250 episódios
    all_runs.append(df)
if not all_runs:
    raise FileNotFoundError("Nenhuma rodada foi carregada. Verifique os arquivos CSV.")

df_all = pd.concat(all_runs, ignore_index=True)


####################################################################################################

# RAW

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="Reward", hue="Run", palette="tab10", linewidth=1.8)
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa')
plt.title("Desempenho do Deep Q-Learning no MountainCar (5 rodagens)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.legend(title="Rodagens")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_dqn_seaborn_linhas.png", dpi=300)
plt.show()

####################################################################################################
# MA 125

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_all, x="Episode", y="Reward_MA", hue="Run", palette="tab10", linewidth=2)
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
    df_all.groupby("Episode")["Reward"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "Reward_Mean", "std": "Reward_SD"})
)
agg["Reward_Mean_Smooth"] = agg["Reward_Mean"].rolling(window=75, min_periods=1).mean()
agg["Reward_SD_Smooth"]   = agg["Reward_SD"].rolling(window=75, min_periods=1).mean()
agg["Upper"] = agg["Reward_Mean_Smooth"] + agg["Reward_SD_Smooth"]
agg["Lower"] = agg["Reward_Mean_Smooth"] - agg["Reward_SD_Smooth"]

plt.figure(figsize=(14, 6))
plt.plot(agg["Episode"], agg["Reward_Mean_Smooth"], label="Recompensa Média (5 rodagens)", color="purple", linewidth=2)
plt.fill_between(agg["Episode"], agg["Lower"], agg["Upper"], color="purple", alpha=0.2, label="±1 Desvio Padrão")
plt.axhline(y=-110, color='black', linestyle='--', linewidth=2.5, label='Meta de Recompensa (-110)')
plt.title("Desempenho Médio do Deep Q-Learning com Variância (MountainCar)")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Média Suavizada")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("images/comparacao_dqn_media_com_desvio.png", dpi=300)
plt.show()

