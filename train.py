import torch
import matplotlib.pyplot as plt
from src.dataset.nor_sir_dataset import SIRDataset

def test_wavelet_reconstruction():

    dataset = SIRDataset(
        params_path="data/raw/params.csv",
        states_path="data/raw/states.csv",
        wavelet="db4",
        level=3,
        j=1,
        normalize=True
    )
    

    # گرفتن اولین run
    # چون dataset transition-based است،
    # ما فقط اولین 100 نمونه را برای یک run می‌گیریم
    #T_plot = 100

    Aj = []
    Dj = []
    Dj1 = []
    original = []

    # برای بازیابی trajectory اصلی،
    # مستقیماً از states.csv بخوان
    import pandas as pd
    states_df = pd.read_csv("data/raw/states.csv")
    run_id = states_df["run_id"].unique()[0]
    run_df = states_df[states_df["run_id"] == run_id].sort_values("t")
    T_plot = len(run_df) - 1   # 👈 این خط مهم است
    

    t = range(T_plot)
    

    original = run_df[["S", "I", "R"]].values[:T_plot]
    mean = original.mean(axis=0)
    std  = original.std(axis=0)

# جلوگیری از تقسیم بر صفر
    std = std + 1e-8

# نرمال‌سازی
    original_norm = (original - mean) / std

    for i in range(T_plot):
        sample = dataset[i]
        Aj.append(sample["Aj_t"].numpy())
        Dj.append(sample["Dj_t"].numpy())
        Dj1.append(sample["Dj1_t"].numpy())

    Aj = torch.tensor(Aj)
    Dj = torch.tensor(Dj)
    Dj1 = torch.tensor(Dj1)

    Aj1 = Aj + Dj
    Aj2 = Aj + Dj + Dj1

   

    # فقط مؤلفه اول (مثلاً S) را رسم می‌کنیم
    dim = 2

    plt.figure(figsize=(12, 8))

    plt.plot(t, original_norm[:, dim], label="Original S", linewidth=2)
    plt.plot(t, Aj[:, dim], label="Aj")
    plt.plot(t, Dj[:, dim], label="Dj")
    plt.plot(t, Dj1[:, dim], label="Dj+1")
    plt.plot(t, Aj1[:, dim], label="Aj+1")
    plt.plot(t, Aj2[:, dim], label="Aj+2")

    plt.legend()
    plt.title("Wavelet Multi-Resolution Sanity Check")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_wavelet_reconstruction()