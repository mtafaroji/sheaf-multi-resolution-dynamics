import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset


class SIRDataset(Dataset):
    """
    Builds transition samples:
    {
        Aj(t), Dj(t), Dj+1(t),
        params,
        Aj(t+1), Dj(t+1), Dj+1(t+1)
    }

    All components are reconstructed time-series
    with the same length as the original trajectory.
    """

    def __init__(
        self,
        params_path: str,
        states_path: str,
        wavelet: str = "db4",
        level: int = 3,
        j: int = 1,  # choose which Aj, Dj to use
    ):

        if not (1 <= j < level):
            raise ValueError("j must satisfy 1 <= j < level")

        self.samples = []

        # ---- Load CSV files ----
        params_df = pd.read_csv(params_path)
        states_df = pd.read_csv(states_path)

        # sort to ensure correct time order
        states_df = states_df.sort_values(["run_id", "t"])

        # ---- Loop over each run ----
        for run_id, run_states_df in states_df.groupby("run_id"):

            run_states_df = run_states_df.sort_values("t")
            states = run_states_df[["S", "I", "R"]].to_numpy(dtype=np.float64)
            T = states.shape[0]

            # ---- Extract parameters for this run ----
            row = params_df[params_df["run_id"] == run_id]
            if len(row) != 1:
                raise ValueError(f"Expected one params row for run_id={run_id}")

            row = row.iloc[0]
            params = np.array([
                row["transmissionProbability"],
                row["meanTimeToRecover"],
                row["meanImmunityDuration"]
            ], dtype=np.float32)

            # Prepare storage for reconstructed signals
            Aj_sig  = np.zeros((T, 3), dtype=np.float32)
            Dj_sig  = np.zeros((T, 3), dtype=np.float32)
            Dj1_sig = np.zeros((T, 3), dtype=np.float32)

            # index mapping for D_list
            # D_list = [D_L, D_{L-1}, ..., D_1]
            Dj_index  = level - j - 1
            Dj1_index = level - j

            for dim in range(3):

                signal = states[:, dim]

                A_L, D_list = self._reconstruct_A_and_Ds(
                    signal, wavelet, level
                )

                # ---- Build Aj = A_L + sum_{k=L down to j+1} D_k ----
                Aj_dim = A_L.copy()
 
                for idx in range(0, Dj_index):   
                    Aj_dim += D_list[idx]
               
                # ---- Build Aj correctly ----



                # ---- Build Dj ----------
                Dj_dim  = D_list[Dj_index]
                Dj1_dim = D_list[Dj1_index]



                Aj_sig[:, dim]  = Aj_dim
                Dj_sig[:, dim]  = Dj_dim
                Dj1_sig[:, dim] = Dj1_dim

            # ---- Build transition samples ----
            for t in range(T - 1):

                self.samples.append({
                    "run_id": run_id,
                    "Aj_t": torch.tensor(Aj_sig[t], dtype=torch.float32),
                    "Dj_t": torch.tensor(Dj_sig[t], dtype=torch.float32),
                    "Dj1_t": torch.tensor(Dj1_sig[t], dtype=torch.float32),
                    "params": torch.tensor(params, dtype=torch.float32),
                    "Aj_tp1": torch.tensor(Aj_sig[t+1], dtype=torch.float32),
                    "Dj_tp1": torch.tensor(Dj_sig[t+1], dtype=torch.float32),
                    "Dj1_tp1": torch.tensor(Dj1_sig[t+1], dtype=torch.float32),
                })

    # -------------------------------------------------
    # -------- Wavelet Reconstruction Helpers ---------
    # -------------------------------------------------

    @staticmethod
    def _waverec_component(coeffs, wavelet, keep_index, length):
        """
        Reconstruct exactly one wavelet component
        (either cA_L or one of cD_k)
        as full-length time-series.
        """
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
        """
        Returns:
            A_L  (coarsest approximation)
            D_list = [D_L, D_{L-1}, ..., D_1]
        All reconstructed to original signal length.
        """

        T = len(signal)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]

        # A_L
        A_L = SIRDataset._waverec_component(
            coeffs, wavelet, keep_index=0, length=T
        )

        # D components
        D_list = []
        for keep_index in range(1, level + 1):
            Dk = SIRDataset._waverec_component(
                coeffs, wavelet, keep_index, T
            )
            D_list.append(Dk)

        return A_L, D_list

    # -------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]