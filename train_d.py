import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


def waverec_component(coeffs, wavelet, keep_idx, T, mode="periodization"):
    new = []
    for i, c in enumerate(coeffs):
        new.append(c if i == keep_idx else np.zeros_like(c))
    rec = pywt.waverec(new, wavelet, mode=mode)

    # center-crop to length T (بهتر از [:T])
    if len(rec) > T:
        start = (len(rec) - T) // 2
        rec = rec[start:start + T]
    elif len(rec) < T:
        pad = T - len(rec)
        rec = np.pad(rec, (pad // 2, pad - pad // 2), mode="edge")
    return rec


def test_single_run_reconstruction():
    wavelet = "db4"
    level = 2
    mode = "periodization"

    states_df = pd.read_csv("data/raw/states.csv")
    run_id = states_df["run_id"].unique()[0]
    run_df = states_df[states_df["run_id"] == run_id].sort_values("t")

    S = np.array(run_df["S"].values, dtype=np.float64)
    #S = run_df["S"].to_numpy()
    T = len(S)

    coeffs = pywt.wavedec(S, wavelet, level=level, mode=mode)  # [cA2, cD2, cD1]

    A2 = waverec_component(coeffs, wavelet, keep_idx=0, T=T, mode=mode)
    D2 = waverec_component(coeffs, wavelet, keep_idx=1, T=T, mode=mode)
    D1 = waverec_component(coeffs, wavelet, keep_idx=2, T=T, mode=mode)

    S_rec = A2 + D2 + D1 

    print("max abs error:", np.max(np.abs(S - S_rec)))
    print("mean abs error:", np.mean(np.abs(S - S_rec)))

    plt.figure(figsize=(12, 6))
    plt.plot(S, label="Original S", linewidth=2)
    plt.plot(A2, label="Aj (=A2)")
    plt.plot(D2, label="Dj (=D2)")
    plt.plot(D1, label="Dj+1 (=D1)")
    plt.plot(S_rec, label="Aj+2 (=A2+D2+D1)", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_single_run_reconstruction()