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
start_test = 80

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * indices_per_run
        end   = start + indices_per_run
        idx.extend(range(start, end))
    return idx

test_idx = run_to_indices(start_test, test_runs)
test_ds = Subset(dataset, test_idx)

test_loader = DataLoader(
    test_ds,
    batch_size=time_length,
    shuffle=False
)

# -----------------------------
# Load models
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
window = 8

total_A = 0
total_D = 0
total_D1 = 0
count = 0

saved_plot_data = None

with torch.no_grad():
    for batch in test_loader:

        Aj_t   = batch["Aj_t"].to(device)
        Dj_t   = batch["Dj_t"].to(device)
        Dj1_t  = batch["Dj1_t"].to(device)

        Aj_tp1  = batch["Aj_tp1"].to(device)
        Dj_tp1  = batch["Dj_tp1"].to(device)
        Dj1_tp1 = batch["Dj1_tp1"].to(device)

        params = batch["params"][0].unsqueeze(0).to(device)

        # -----------------------------
        # Window-based loss
        # -----------------------------
        for t in range(time_length - window):

            Aj_current  = Aj_t[t]
            Dj_current  = Dj_t[t]
            Dj1_current = Dj1_t[t]

            for k in range(window):

                pred_A  = net_A(Aj_current.unsqueeze(0),  params).squeeze(0)
                pred_D  = net_D(Dj_current.unsqueeze(0),  params).squeeze(0)
                pred_D1 = net_D1(Dj1_current.unsqueeze(0), params).squeeze(0)

                total_A  += criterion(pred_A,  Aj_tp1[t+k]).item()
                total_D  += criterion(pred_D,  Dj_tp1[t+k]).item()
                total_D1 += criterion(pred_D1, Dj1_tp1[t+k]).item()

                Aj_current  = pred_A
                Dj_current  = pred_D
                Dj1_current = pred_D1

                count += 1

        # -----------------------------
        # Reconstruction for plotting
        # -----------------------------
        if saved_plot_data is None:

            Aj_pred_full  = torch.zeros_like(Aj_tp1)
            Dj_pred_full  = torch.zeros_like(Dj_tp1)
            Dj1_pred_full = torch.zeros_like(Dj1_tp1)

            for t in range(time_length):

                if t % window == 0:
                    Aj_current  = Aj_t[t]
                    Dj_current  = Dj_t[t]
                    Dj1_current = Dj1_t[t]

                pred_A  = net_A(Aj_current.unsqueeze(0),  params).squeeze(0)
                pred_D  = net_D(Dj_current.unsqueeze(0),  params).squeeze(0)
                pred_D1 = net_D1(Dj1_current.unsqueeze(0), params).squeeze(0)

                Aj_pred_full[t]  = pred_A
                Dj_pred_full[t]  = pred_D
                Dj1_pred_full[t] = pred_D1

                Aj_current  = pred_A
                Dj_current  = pred_D
                Dj1_current = pred_D1

            saved_plot_data = {
                "Aj_true": Aj_tp1.cpu(),
                "Aj_pred": Aj_pred_full.cpu(),
                "Dj_true": Dj_tp1.cpu(),
                "Dj_pred": Dj_pred_full.cpu(),
                "Dj1_true": Dj1_tp1.cpu(),
                "Dj1_pred": Dj1_pred_full.cpu(),
            }

        break  # فقط یک ران برای رسم

print("\nTest Window-Based Loss:")
print("A  :", total_A / count)
print("D  :", total_D / count)
print("D1 :", total_D1 / count)

# -----------------------------
# Plot
# -----------------------------
state_index = 1
t_axis = range(time_length)

plt.figure()
plt.plot(t_axis, saved_plot_data["Aj_true"][:, state_index], label="True")
plt.plot(t_axis, saved_plot_data["Aj_pred"][:, state_index], label="Pred")
plt.title("Aj Reconstruction - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_axis, saved_plot_data["Dj_true"][:, state_index], label="True")
plt.plot(t_axis, saved_plot_data["Dj_pred"][:, state_index], label="Pred")
plt.title("Dj Reconstruction - State I")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_axis, saved_plot_data["Dj1_true"][:, state_index], label="True")
plt.plot(t_axis, saved_plot_data["Dj1_pred"][:, state_index], label="Pred")
plt.title("Dj1 Reconstruction - State I")
plt.legend()
plt.show()