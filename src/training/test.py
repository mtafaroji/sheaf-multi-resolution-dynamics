import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.dataset.sir_dataset import SIRDataset
from src.models.transition_mlp import TransitionMLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Dataset
# --------------------------------------------------
dataset = SIRDataset(
    params_path="data/raw/params.csv",
    states_path="data/raw/states.csv",
    wavelet="db4",
    level=3,
    j=1,
)

import pandas as pd

states_df = pd.read_csv("data/raw/states.csv")

# طول واقعی یک run
first_run = states_df["run_id"].unique()[0]
run_df = states_df[states_df["run_id"] == first_run]

time_length = len(run_df) - 1   # چون dataset transition-based است
indices_per_run = time_length

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

# test = آخرین 20 ران
test_idx = run_to_indices(80, 20)
test_ds = Subset(dataset, test_idx)

# batch_size = کل trajectory یک run
test_loader = DataLoader(test_ds, batch_size=time_length, shuffle=False)

# --------------------------------------------------
# Load models
# --------------------------------------------------
net_A  = TransitionMLP().to(device)
net_D  = TransitionMLP().to(device)
net_D1 = TransitionMLP().to(device)

net_A.load_state_dict(torch.load("best_A.pt", map_location=device))
net_D.load_state_dict(torch.load("best_D.pt", map_location=device))
net_D1.load_state_dict(torch.load("best_D1.pt", map_location=device))

net_A.eval()
net_D.eval()
net_D1.eval()

criterion = nn.MSELoss()

# --------------------------------------------------
# Rollout Evaluation
# --------------------------------------------------
with torch.no_grad():

    for batch in test_loader:

        # ----------------------------
        # Extract single run data
        # ----------------------------
        Aj_t   = batch["Aj_t"].to(device)      # (T,3)
        Dj_t   = batch["Dj_t"].to(device)
        Dj1_t  = batch["Dj1_t"].to(device)

        Aj_tp1  = batch["Aj_tp1"].to(device)   # ground truth
        Dj_tp1  = batch["Dj_tp1"].to(device)
        Dj1_tp1 = batch["Dj1_tp1"].to(device)

        params_all = batch["params"].to(device)   # (T,3)

        # چون پارامتر در کل run ثابت است:
        params = params_all[0].unsqueeze(0)       # (1,3)

        # ----------------------------
        # Initial condition
        # ----------------------------
        Aj_current  = Aj_t[0]    # (3,)
        Dj_current  = Dj_t[0]
        Dj1_current = Dj1_t[0]

        Aj_preds  = []
        Dj_preds  = []
        Dj1_preds = []

        # ----------------------------
        # True Rollout
        # ----------------------------
        for t in range(time_length):

            xA  = Aj_current.unsqueeze(0)   # (1,3)
            xD  = Dj_current.unsqueeze(0)
            xD1 = Dj1_current.unsqueeze(0)

            next_A  = net_A(xA,  params).squeeze(0)
            next_D  = net_D(xD,  params).squeeze(0)
            next_D1 = net_D(xD1, params).squeeze(0)

            Aj_preds.append(next_A)
            Dj_preds.append(next_D)
            Dj1_preds.append(next_D1)

            # autoregressive update
            #l = t + 1   # چون Aj_tp1 و بقیه از t=1 شروع می‌شوند
            if t < time_length -1 :   # تا قبل از آخرین گام
                Aj_current  = Aj_tp1[t+1]
                Dj_current  = Dj_tp1[t+1]
                Dj1_current = Dj1_tp1[t+1]


        Aj_preds  = torch.stack(Aj_preds)
        Dj_preds  = torch.stack(Dj_preds)
        Dj1_preds = torch.stack(Dj1_preds)

        # فقط اولین trajectory را بررسی می‌کنیم
        break


# --------------------------------------------------
# Compute Rollout Loss
# --------------------------------------------------
loss_A  = criterion(Aj_preds,  Aj_tp1).item()
loss_D  = criterion(Dj_preds,  Dj_tp1).item()
loss_D1 = criterion(Dj1_preds, Dj1_tp1).item()

print("\nRollout Test Loss:")
print("A  :", loss_A)
print("D  :", loss_D)
print("D1 :", loss_D1)


# --------------------------------------------------
# Plot One State (مثلاً I)
# --------------------------------------------------
state_index = 1   # 0=S, 1=I, 2=R
t = range(time_length)

plt.figure()
plt.plot(t, Aj_tp1[:, state_index].cpu(), label="True")
plt.plot(t, Aj_preds[:, state_index].cpu(), label="Rollout")
plt.title("Aj Rollout - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, Dj_tp1[:, state_index].cpu(), label="True")
plt.plot(t, Dj_preds[:, state_index].cpu(), label="Rollout")
plt.title("Dj Rollout - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, Dj1_tp1[:, state_index].cpu(), label="True")
plt.plot(t, Dj1_preds[:, state_index].cpu(), label="Rollout")
plt.title("Dj1 Rollout - State I")
plt.legend()
plt.show()