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

# طول واقعی یک run
first_run = states_df["run_id"].unique()[0]
run_df = states_df[states_df["run_id"] == first_run]

time_length = len(run_df) - 1   # چون dataset transition-based است











indices_per_run = time_length






test_runs = 20
start_test = 80  # آخرین 20 ران

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

test_idx = run_to_indices(start_test, test_runs)
test_ds = Subset(dataset, test_idx)

runs_per_batch = 8
batch_size = runs_per_batch * time_length

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)

# -----------------------------
# Load Models
# -----------------------------
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
window = 5

total_A = 0
total_D = 0
total_D1 = 0
count = 0

# برای رسم یک ران نگه می‌داریم
saved_plot_data = None

with torch.no_grad():
    for batch in test_loader:

        current_batch_size = batch["Aj_t"].shape[0] // time_length

        Aj_t = batch["Aj_t"].view(current_batch_size, time_length, 3).to(device)
        Dj_t = batch["Dj_t"].view(current_batch_size, time_length, 3).to(device)
        Dj1_t = batch["Dj1_t"].view(current_batch_size, time_length, 3).to(device)

        Aj_tp1 = batch["Aj_tp1"].view(current_batch_size, time_length, 3).to(device)
        Dj_tp1 = batch["Dj_tp1"].view(current_batch_size, time_length, 3).to(device)
        Dj1_tp1 = batch["Dj1_tp1"].view(current_batch_size, time_length, 3).to(device)

        params = batch["params"].view(current_batch_size, time_length, 3)[:,0,:].to(device)

        for r in range(current_batch_size):

            Aj_current = Aj_t[r,0,:]
            Dj_current = Dj_t[r,0,:]
            Dj1_current = Dj1_t[r,0,:]

            Aj_preds = []
            Dj_preds = []
            Dj1_preds = []

            for t in range(time_length):

                pred_A  = net_A(Aj_current.unsqueeze(0),  params[r].unsqueeze(0)).squeeze(0)
                pred_D  = net_D(Dj_current.unsqueeze(0),  params[r].unsqueeze(0)).squeeze(0)
                pred_D1 = net_D1(Dj1_current.unsqueeze(0), params[r].unsqueeze(0)).squeeze(0)

                Aj_preds.append(pred_A)
                Dj_preds.append(pred_D)
                Dj1_preds.append(pred_D1)

                Aj_current = pred_A
                Dj_current = pred_D
                Dj1_current = pred_D1

            Aj_preds = torch.stack(Aj_preds)
            Dj_preds = torch.stack(Dj_preds)
            Dj1_preds = torch.stack(Dj1_preds)

            total_A += criterion(Aj_preds, Aj_tp1[r]).item()
            total_D += criterion(Dj_preds, Dj_tp1[r]).item()
            total_D1 += criterion(Dj1_preds, Dj1_tp1[r]).item()
            count += 1

            if saved_plot_data is None:
                saved_plot_data = {
                    "Aj_true": Aj_tp1[r].cpu(),
                    "Aj_pred": Aj_preds.cpu(),
                    "Dj_true": Dj_tp1[r].cpu(),
                    "Dj_pred": Dj_preds.cpu(),
                    "Dj1_true": Dj1_tp1[r].cpu(),
                    "Dj1_pred": Dj1_preds.cpu(),
                }

print("\nTest Multi-Step Rollout Loss:")
print("A  :", total_A / count)
print("D  :", total_D / count)
print("D1 :", total_D1 / count)

# -----------------------------
# Plot One State (مثلاً I)
# -----------------------------
state_index = 1
t = range(time_length)

plt.figure()
plt.plot(t, saved_plot_data["Aj_true"][:, state_index], label="True")
plt.plot(t, saved_plot_data["Aj_pred"][:, state_index], label="Pred")
plt.title("Aj Reconstruction - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, saved_plot_data["Dj_true"][:, state_index], label="True")
plt.plot(t, saved_plot_data["Dj_pred"][:, state_index], label="Pred")
plt.title("Dj Reconstruction - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t, saved_plot_data["Dj1_true"][:, state_index], label="True")
plt.plot(t, saved_plot_data["Dj1_pred"][:, state_index], label="Pred")
plt.title("Dj1 Reconstruction - State I")
plt.legend()
plt.show()