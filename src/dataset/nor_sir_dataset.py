import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset


class SIRDataset(Dataset):

    def __init__(
        self,
        params_path: str,
        states_path: str,
        wavelet: str = "db4",
        level: int = 3,
        j: int = 1,
        normalize: bool = True,
        stats: dict = None,   # اگر بخواهی mean/std آماده بدهی (برای val/test)
    ):

        if not (1 <= j < level):
            raise ValueError("j must satisfy 1 <= j < level")

        self.samples = []
        self.normalize = normalize

        params_df = pd.read_csv(params_path)
        states_df = states_df = pd.read_csv(states_path)
        states_df = states_df.sort_values(["run_id", "t"])

        all_Aj = []
        all_Dj = []
        all_Dj1 = []

        for run_id, run_states_df in states_df.groupby("run_id"):

            run_states_df = run_states_df.sort_values("t")
            states = run_states_df[["S", "I", "R"]].to_numpy(dtype=np.float64)
            T = states.shape[0]

            row = params_df[params_df["run_id"] == run_id].iloc[0]

            params = np.array([
                row["transmissionProbability"],
                row["meanTimeToRecover"],
                row["meanImmunityDuration"]
            ], dtype=np.float32)

            Aj_sig  = np.zeros((T, 3), dtype=np.float32)
            Dj_sig  = np.zeros((T, 3), dtype=np.float32)
            Dj1_sig = np.zeros((T, 3), dtype=np.float32)

            Dj_index  = level - j - 1
            Dj1_index = level - j

            for dim in range(3):

                signal = states[:, dim]
                A_L, D_list = self._reconstruct_A_and_Ds(signal, wavelet, level)

                Aj_dim = A_L.copy()
                for idx in range(0, Dj_index):
                    Aj_dim += D_list[idx]

                Dj_dim  = D_list[Dj_index]
                Dj1_dim = D_list[Dj1_index]

                Aj_sig[:, dim]  = Aj_dim
                Dj_sig[:, dim]  = Dj_dim
                Dj1_sig[:, dim] = Dj1_dim

            # collect for normalization statistics
            all_Aj.append(Aj_sig)
            all_Dj.append(Dj_sig)
            all_Dj1.append(Dj1_sig)

            for t in range(T - 1):
                self.samples.append({
                    "run_id": run_id,
                    "Aj_t": Aj_sig[t],
                    "Dj_t": Dj_sig[t],
                    "Dj1_t": Dj1_sig[t],
                    "params": params,
                    "Aj_tp1": Aj_sig[t+1],
                    "Dj_tp1": Dj_sig[t+1],
                    "Dj1_tp1": Dj1_sig[t+1],
                })

        # -----------------------------
        # Compute normalization stats
        # -----------------------------
        if normalize:

            if stats is None:
                Aj_all = np.concatenate(all_Aj, axis=0)
                Dj_all = np.concatenate(all_Dj, axis=0)
                Dj1_all = np.concatenate(all_Dj1, axis=0)

                self.stats = {
                    "Aj_mean": Aj_all.mean(axis=0),
                    "Aj_std":  Aj_all.std(axis=0) + 1e-8,
                    "Dj_mean": Dj_all.mean(axis=0),
                    "Dj_std":  Dj_all.std(axis=0) + 1e-8,
                    "Dj1_mean": Dj1_all.mean(axis=0),
                    "Dj1_std":  Dj1_all.std(axis=0) + 1e-8,
                }
            else:
                self.stats = stats

            # Apply normalization
            for s in self.samples:
                s["Aj_t"]   = (s["Aj_t"]   - self.stats["Aj_mean"])  / self.stats["Aj_std"]
                s["Aj_tp1"] = (s["Aj_tp1"] - self.stats["Aj_mean"])  / self.stats["Aj_std"]

                s["Dj_t"]   = (s["Dj_t"]   - self.stats["Dj_mean"])  / self.stats["Dj_std"]
                s["Dj_tp1"] = (s["Dj_tp1"] - self.stats["Dj_mean"])  / self.stats["Dj_std"]

                s["Dj1_t"]   = (s["Dj1_t"]   - self.stats["Dj1_mean"]) / self.stats["Dj1_std"]
                s["Dj1_tp1"] = (s["Dj1_tp1"] - self.stats["Dj1_mean"]) / self.stats["Dj1_std"]

        # finally convert to torch
        for s in self.samples:
            for k in s:
                if k != "run_id":
                    s[k] = torch.tensor(s[k], dtype=torch.float32)

    # -------------------------------------------------

    @staticmethod
    def _waverec_component(coeffs, wavelet, keep_index, length):
        new_coeffs = []
        for i, c in enumerate(coeffs):
            if i == keep_index:
                new_coeffs.append(c)
            else:
                new_coeffs.append(np.zeros_like(c))
        rec = pywt.waverec(new_coeffs, wavelet)
        return rec[:length]

    @staticmethod
    def _reconstruct_A_and_Ds(signal, wavelet="db4", level=3):
        T = len(signal)
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        A_L = SIRDataset._waverec_component(coeffs, wavelet, 0, T)

        D_list = []
        for keep_index in range(1, level + 1):
            Dk = SIRDataset._waverec_component(coeffs, wavelet, keep_index, T)
            D_list.append(Dk)

        return A_L, D_list

    # -------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]