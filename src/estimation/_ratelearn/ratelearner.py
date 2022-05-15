import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .rate import RateMatrix
from .trainer import train_quantization


def solve_stationery_dist(rate_matrix):
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist


def normalized(Q):
    pi = solve_stationery_dist(Q)
    mutation_rate = pi @ -np.diag(Q)
    return Q / mutation_rate


class RateMatrixLearner:
    def __init__(
        self,
        branches: List[float],
        mats: List[np.array],
        output_dir: str,
        stationnary_distribution: str,
        device: str,
        mask: str = None,
        rate_matrix_parameterization="pande_reversible",
        initialization: Optional[np.array] = None,
    ):
        self.branches = branches
        self.mats = mats
        self.output_dir = output_dir
        self.stationnary_distribution = stationnary_distribution
        self.mask = mask
        self.rate_matrix_parameterization = rate_matrix_parameterization
        self.lr = None
        self.do_adam = None
        self.df_res = None
        self.Qfinal = None
        self.trained_ = False
        self.device = device
        self.initialization = initialization
        self._learned_matrix_path = os.path.join(
            self.output_dir, "learned_matrix.txt"
        )
        self._normalized_learned_matrix_path = os.path.join(
            self.output_dir, "learned_matrix_normalized.txt"
        )
        self._df_res_filepath = os.path.join(
            self.output_dir, "training_df.pickle"
        )
        self._figpath = os.path.join(self.output_dir, "training_plot.pdf")

    def train(
        self,
        lr=1e-1,
        num_epochs=2000,
        do_adam: bool = True,
    ):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting, outdir: {self.output_dir}")

        torch.manual_seed(0)
        device = self.device
        output_dir = self.output_dir
        initialization = self.initialization

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open frequency matrices
        self.quantized_data, self.n_states = self.get_branch_to_mat()

        # Open stationnary distribution if necessary
        pi_path = self.stationnary_distribution
        if pi_path is not None:
            pi = pd.read_csv(
                pi_path, header=None, index_col=None
            ).values.squeeze()
        else:
            pi = np.ones(self.n_states)
            pi = pi / pi.sum()
        self.pi = torch.tensor(pi).float()

        mask_path = self.mask
        if mask_path is not None:
            mask_mat = pd.read_csv(
                mask_path, sep="\s", header=None, index_col=None
            ).values
        else:
            mask_mat = np.ones((self.n_states, self.n_states))
        mask_mat = torch.tensor(mask_mat, dtype=torch.float)
        self.mask_mat = mask_mat

        pi_requires_grad = pi_path is None
        self.mat_module = RateMatrix(
            num_states=self.n_states,
            mode=self.rate_matrix_parameterization,
            pi=self.pi,
            pi_requires_grad=pi_requires_grad,
            initialization=initialization,
            mask=mask_mat,
        ).to(device=device)

        self.lr = lr
        self.do_adam = do_adam

        if self.do_adam:
            optim = torch.optim.Adam(
                params=self.mat_module.parameters(), lr=self.lr
            )
        else:
            optim = torch.optim.SGD(
                params=self.mat_module.parameters(), lr=self.lr
            )

        df_res, Q = train_quantization(
            rate_module=self.mat_module,
            quantized_dataset=self.quantized_data,
            num_epochs=num_epochs,
            Q_true=None,
            optimizer=optim,
        )
        self.df_res = df_res
        self.Qfinal = Q
        self.trained = True
        self.process_results()

    def get_branch_to_mat(self):
        n_features = self.mats[0].shape[0]
        qtimes = torch.tensor(self.branches)
        cmats = torch.tensor(self.mats)
        quantized_data = TensorDataset(qtimes, cmats)
        return quantized_data, n_features

    def process_results(self):
        learned_matrix_path = self._learned_matrix_path
        Q = self.Qfinal.detach().cpu().numpy()
        np.savetxt(learned_matrix_path, Q)
        os.system(f"chmod 555 {learned_matrix_path}")

        normalized_learned_matrix_path = self._normalized_learned_matrix_path
        np.savetxt(normalized_learned_matrix_path, normalized(Q))
        os.system(f"chmod 555 {normalized_learned_matrix_path}")

        df_res_filepath = self._df_res_filepath
        self.df_res.to_pickle(df_res_filepath)
        os.system(f"chmod 555 {df_res_filepath}")

        figpath = self._figpath
        FT_SIZE = 13
        fig, axes = plt.subplots(figsize=(5, 4))
        self.df_res.loss.plot()
        plt.xscale("log")
        plt.ylabel("Negative likelihood", fontsize=FT_SIZE)
        plt.xlabel("# of iterations", fontsize=FT_SIZE)
        plt.tight_layout()
        plt.savefig(figpath)
        os.system(f"chmod 555 {figpath}")
