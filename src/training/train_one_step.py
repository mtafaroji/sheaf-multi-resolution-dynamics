import torch
import torch.nn as nn
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



# هر run شامل time_length سطر است

indices_per_run = time_length

# تقسیم run ها
train_runs = 64 * runs // 100
val_runs   = 16 * runs // 100
test_runs  = 20 * runs // 100

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

train_idx = run_to_indices(0, train_runs)
val_idx   = run_to_indices(train_runs, val_runs)
test_idx  = run_to_indices(train_runs + val_runs, test_runs)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)
test_ds  = Subset(dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=time_length, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=time_length, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=time_length, shuffle=False)

# -----------------------------
# Models
# -----------------------------
net_A  = TransitionMLP().to(device)
net_D  = TransitionMLP().to(device)
net_D1 = TransitionMLP().to(device)

opt_A  = torch.optim.Adam(net_A.parameters(),  lr=1e-3)
opt_D  = torch.optim.Adam(net_D.parameters(),  lr=1e-3)
opt_D1 = torch.optim.Adam(net_D1.parameters(), lr=1e-3)

criterion = nn.MSELoss()

best_val_A  = float("inf")
best_val_D  = float("inf")
best_val_D1 = float("inf")

epochs = 50

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    net_A.train()
    net_D.train()
    net_D1.train()

    for batch in train_loader:

        Aj_t   = batch["Aj_t"].to(device)
        Dj_t   = batch["Dj_t"].to(device)
        Dj1_t  = batch["Dj1_t"].to(device)

        Aj_tp1  = batch["Aj_tp1"].to(device)
        Dj_tp1  = batch["Dj_tp1"].to(device)
        Dj1_tp1 = batch["Dj1_tp1"].to(device)

        params = batch["params"].to(device)

        # ---- A network ----
        pred_A = net_A(Aj_t, params)
        loss_A = criterion(pred_A, Aj_tp1)
        opt_A.zero_grad()
        loss_A.backward()
        opt_A.step()

        # ---- D network ----
        pred_D = net_D(Dj_t, params)
        loss_D = criterion(pred_D, Dj_tp1)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- D1 network ----
        pred_D1 = net_D1(Dj1_t, params)
        loss_D1 = criterion(pred_D1, Dj1_tp1)
        opt_D1.zero_grad()
        loss_D1.backward()
        opt_D1.step()

    # -----------------------------
    # Validation
    # -----------------------------
    net_A.eval()
    net_D.eval()
    net_D1.eval()

    val_loss_A  = 0
    val_loss_D  = 0
    val_loss_D1 = 0

    with torch.no_grad():
        for batch in val_loader:

            Aj_t   = batch["Aj_t"].to(device)
            Dj_t   = batch["Dj_t"].to(device)
            Dj1_t  = batch["Dj1_t"].to(device)

            Aj_tp1  = batch["Aj_tp1"].to(device)
            Dj_tp1  = batch["Dj_tp1"].to(device)
            Dj1_tp1 = batch["Dj1_tp1"].to(device)

            params = batch["params"].to(device)

            val_loss_A  += criterion(net_A(Aj_t, params),  Aj_tp1).item()
            val_loss_D  += criterion(net_D(Dj_t, params),  Dj_tp1).item()
            val_loss_D1 += criterion(net_D1(Dj1_t, params), Dj1_tp1).item()

    val_loss_A  /= len(val_loader)
    val_loss_D  /= len(val_loader)
    val_loss_D1 /= len(val_loader)

    print(f"Epoch {epoch+1} | "
          f"Val A={val_loss_A:.6f} "
          f"Val D={val_loss_D:.6f} "
          f"Val D1={val_loss_D1:.6f}")

    # -----------------------------
    # Save best per resolution
    # -----------------------------
    if val_loss_A < best_val_A:
        best_val_A = val_loss_A
        torch.save(net_A.state_dict(), "best_A.pt")

    if val_loss_D < best_val_D:
        best_val_D = val_loss_D
        torch.save(net_D.state_dict(), "best_D.pt")

    if val_loss_D1 < best_val_D1:
        best_val_D1 = val_loss_D1
        torch.save(net_D1.state_dict(), "best_D1.pt")

print("Training finished.")