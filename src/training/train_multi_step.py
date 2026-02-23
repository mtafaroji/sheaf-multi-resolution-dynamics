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
# طول واقعی یک run
first_run = states_df["run_id"].unique()[0]
run_df = states_df[states_df["run_id"] == first_run]

time_length = len(run_df) - 1 
runs = len(states_df["run_id"].unique())

indices_per_run = time_length
train_runs = 64 * runs // 100
val_runs   = 16 * runs // 100

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

train_idx = run_to_indices(0, train_runs)
val_idx   = run_to_indices(train_runs, val_runs)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

# -----------------------------
# Batch روی ران‌ها
# -----------------------------
runs_per_batch = 8
batch_size = runs_per_batch * time_length

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

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

window = 8
epochs = 50

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    net_A.train()
    net_D.train()
    net_D1.train()

    for batch in train_loader:

        Aj_t   = batch["Aj_t"].view(runs_per_batch, time_length, 3).to(device)
        Dj_t   = batch["Dj_t"].view(runs_per_batch, time_length, 3).to(device)
        Dj1_t  = batch["Dj1_t"].view(runs_per_batch, time_length, 3).to(device)

        Aj_tp1  = batch["Aj_tp1"].view(runs_per_batch, time_length, 3).to(device)
        Dj_tp1  = batch["Dj_tp1"].view(runs_per_batch, time_length, 3).to(device)
        Dj1_tp1 = batch["Dj1_tp1"].view(runs_per_batch, time_length, 3).to(device)

        params = batch["params"].view(runs_per_batch, time_length, 3)[:,0,:].to(device)

        total_loss_A  = 0
        total_loss_D  = 0
        total_loss_D1 = 0
        count = 0

        for t in range(time_length - window):

            Aj_current  = Aj_t[:, t, :]
            Dj_current  = Dj_t[:, t, :]
            Dj1_current = Dj1_t[:, t, :]

            for k in range(window):

                pred_A  = net_A(Aj_current,  params)
                pred_D  = net_D(Dj_current,  params)
                pred_D1 = net_D1(Dj1_current, params)

                total_loss_A  += criterion(pred_A,  Aj_tp1[:, t+k, :])
                total_loss_D  += criterion(pred_D,  Dj_tp1[:, t+k, :])
                total_loss_D1 += criterion(pred_D1, Dj1_tp1[:, t+k, :])

                Aj_current  = pred_A
                Dj_current  = pred_D
                Dj1_current = pred_D1

                count += 1

        total_loss_A  /= count
        total_loss_D  /= count
        total_loss_D1 /= count
        opt_A.zero_grad()
        total_loss_A.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(net_A.parameters(), 1.0)
        opt_A.step()

        opt_D.zero_grad()
        total_loss_D.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(net_D.parameters(), 1.0)
        opt_D.step()

        opt_D1.zero_grad()
        total_loss_D1.backward()
        torch.nn.utils.clip_grad_norm_(net_D1.parameters(), 1.0)
        opt_D1.step()


    # -----------------------------
    # Validation
    # -----------------------------
    net_A.eval()
    net_D.eval()
    net_D1.eval()

    val_A = 0
    val_D = 0
    val_D1 = 0
    count_val = 0

    with torch.no_grad():
        for batch in val_loader:

            Aj_t   = batch["Aj_t"].view(runs_per_batch, time_length, 3).to(device)
            Dj_t   = batch["Dj_t"].view(runs_per_batch, time_length, 3).to(device)
            Dj1_t  = batch["Dj1_t"].view(runs_per_batch, time_length, 3).to(device)

            Aj_tp1  = batch["Aj_tp1"].view(runs_per_batch, time_length, 3).to(device)
            Dj_tp1  = batch["Dj_tp1"].view(runs_per_batch, time_length, 3).to(device)
            Dj1_tp1 = batch["Dj1_tp1"].view(runs_per_batch, time_length, 3).to(device)

            params = batch["params"].view(runs_per_batch, time_length, 3)[:,0,:].to(device)

            for t in range(time_length - window):

                Aj_current  = Aj_t[:, t, :]
                Dj_current  = Dj_t[:, t, :]
                Dj1_current = Dj1_t[:, t, :]

                for k in range(window):

                    pred_A  = net_A(Aj_current,  params)
                    pred_D  = net_D(Dj_current,  params)
                    pred_D1 = net_D1(Dj1_current, params)

                    val_A  += criterion(pred_A,  Aj_tp1[:, t+k, :]).item()
                    val_D  += criterion(pred_D,  Dj_tp1[:, t+k, :]).item()
                    val_D1 += criterion(pred_D1, Dj1_tp1[:, t+k, :]).item()

                    Aj_current  = pred_A
                    Dj_current  = pred_D
                    Dj1_current = pred_D1

                    count_val += 1

    val_A  /= count_val
    val_D  /= count_val
    val_D1 /= count_val

    print(f"Epoch {epoch+1} | "
          f"Val A={val_A:.6f} "
          f"Val D={val_D:.6f} "
          f"Val D1={val_D1:.6f}")

    # Save best separately
    if val_A < best_val_A:
        best_val_A = val_A
        torch.save(net_A.state_dict(), "best_A.pt")

    if val_D < best_val_D:
        best_val_D = val_D
        torch.save(net_D.state_dict(), "best_D.pt")

    if val_D1 < best_val_D1:
        best_val_D1 = val_D1
        torch.save(net_D1.state_dict(), "best_D1.pt")

print("Training finished.")