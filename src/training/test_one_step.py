import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.dataset.sir_dataset import SIRDataset
from src.models.transition_mlp import TransitionMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------
dataset = SIRDataset(
    params_path="data/raw/params.csv",
    states_path="data/raw/states.csv",
    wavelet="db4",
    level=3,
    j=1,
)


import pandas as pd

states_df = pd.read_csv("data/raw/states.csv")
#paras_df = pd.read_csv("data/raw/params.csv")
# طول واقعی یک run
first_run = states_df["run_id"].unique()[0]
run_df = states_df[states_df["run_id"] == first_run]

time_length = len(run_df) - 1 
#run_inx = run_df["run_id"].iloc[0]
runs = len(states_df["run_id"].unique()) 

print(f'number of runs: {runs}')


start_test = 80 * runs // 100  # شروع آخرین 20 درصد
num_test = 20 * runs // 100    # تعداد run های تست

#runs = 100
indices_per_run = time_length

# test runs = last 20
def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

test_idx = run_to_indices(start_test, num_test)
test_ds = Subset(dataset, test_idx)

test_loader = DataLoader(test_ds, batch_size=time_length, shuffle=False)

# -----------------------------
# Load models
# -----------------------------
net_A  = TransitionMLP().to(device)
net_D  = TransitionMLP().to(device)
net_D1 = TransitionMLP().to(device)

net_A.load_state_dict(torch.load("best_A.pt"))
net_D.load_state_dict(torch.load("best_D.pt"))
net_D1.load_state_dict(torch.load("best_D1.pt"))

net_A.eval()
net_D.eval()
net_D1.eval()

criterion = nn.MSELoss()

total_A  = 0
total_D  = 0
total_D1 = 0

# -----------------------------
# Evaluation
# -----------------------------
with torch.no_grad():
    for batch in test_loader:

        Aj_t   = batch["Aj_t"].to(device)
        Dj_t   = batch["Dj_t"].to(device)
        Dj1_t  = batch["Dj1_t"].to(device)

        Aj_tp1  = batch["Aj_tp1"].to(device)
        Dj_tp1  = batch["Dj_tp1"].to(device)
        Dj1_tp1 = batch["Dj1_tp1"].to(device)

        params = batch["params"].to(device)

        pred_A  = net_A(Aj_t, params)
        pred_D  = net_D(Dj_t, params)
        pred_D1 = net_D1(Dj1_t, params)

        total_A  += criterion(pred_A,  Aj_tp1).item()
        total_D  += criterion(pred_D,  Dj_tp1).item()
        total_D1 += criterion(pred_D1, Dj1_tp1).item()

        # فقط اولین trajectory را برای رسم نگه می‌داریم
        plot_data = {
            "Aj_true": Aj_tp1.cpu(),
            "Aj_pred": pred_A.cpu(),
            "Dj_true": Dj_tp1.cpu(),
            "Dj_pred": pred_D.cpu(),
            "Dj1_true": Dj1_tp1.cpu(),
            "Dj1_pred": pred_D1.cpu(),
        }

        break  # فقط یک trajectory برای رسم

print("Test Loss:")
print("A  :", total_A / len(test_loader))
print("D  :", total_D / len(test_loader))
print("D1 :", total_D1 / len(test_loader))


# -----------------------------
# Plot one state (مثلاً I = index 1)
# -----------------------------
state_index = 1  # 0=S, 1=I, 2=R
t = range(time_length)

# --- Aj ---
plt.figure()
plt.plot(t, plot_data["Aj_true"][:, state_index], label="True")
plt.plot(t, plot_data["Aj_pred"][:, state_index], label="Pred")
plt.title("Aj - State I")
plt.legend()
plt.show()

# --- Dj ---
plt.figure()
plt.plot(t, plot_data["Dj_true"][:, state_index], label="True")
plt.plot(t, plot_data["Dj_pred"][:, state_index], label="Pred")
plt.title("Dj - State I")
plt.legend()
plt.show()

# --- Dj1 ---
plt.figure()
plt.plot(t, plot_data["Dj1_true"][:, state_index], label="True")
plt.plot(t, plot_data["Dj1_pred"][:, state_index], label="Pred")
plt.title("Dj1 - State I")
plt.legend()
plt.show()