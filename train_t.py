import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset.sir_dataset import SIRDataset



def test_dataset_reconstruction():

    ds = SIRDataset(
        params_path="data/raw/params.csv",
        states_path="data/raw/states.csv",
        wavelet="db4",
        level=3,
        j=1
    )

    # ---- پیدا کردن اولین run_id ----
    first_run = ds[0]["run_id"]

    Aj = []
    Dj = []
    Dj1 = []

    # ---- جمع کردن تمام timestep های همان run ----
    for sample in ds:
        if sample["run_id"] == first_run:
            Aj.append(sample["Aj_t"])
            Dj.append(sample["Dj_t"])
            Dj1.append(sample["Dj1_t"])
        else:
            break  # وقتی run عوض شد، توقف

    Aj = torch.stack(Aj)
    Dj = torch.stack(Dj)
    Dj1 = torch.stack(Dj1)

    S_rec = Aj + Dj + Dj1  # باید برابر سیگنال اصلی باشد

    # ---- گرفتن سیگنال اصلی از CSV ----
    states_df = pd.read_csv("data/raw/states.csv")
    run_df = states_df[states_df["run_id"] == first_run].sort_values("t")
    S_original = torch.tensor(
        run_df["S"].values[:len(S_rec)],
        dtype=torch.float32
    )

    # ---- محاسبه خطا ----
    max_err = torch.max(torch.abs(S_original - S_rec[:, 0])).item()
    mean_err = torch.mean(torch.abs(S_original - S_rec[:, 0])).item()

    print("Dataset reconstruction check")
    print("Max abs error :", max_err)
    print("Mean abs error:", mean_err)

    # ---- رسم برای اطمینان دیداری ----
    t = range(len(S_rec))

    plt.figure(figsize=(12, 6))
    plt.plot(t, S_original, label="Original S", linewidth=2)
    plt.plot(t, S_rec[:, 0], label="Aj+Dj+Dj+1", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.title("Dataset Reconstruction Sanity Check")
    plt.show()


if __name__ == "__main__":
    test_dataset_reconstruction()