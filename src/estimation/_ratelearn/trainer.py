import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_sgd(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    num_epochs=10,
    Q_true=None,
    optimizer=None,
    max_iter=1e6,
    batch_size=1000,
    n_every=10,
):
    """
    Quantization baseline
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(rate_module.parameters(), lr=lr)

    print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=batch_size, shuffle=True)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    num_states = rate_module.num_states
    it = 0
    try:
        for epoch in rg:
            # Now compute the loss
            for _, datapoint in enumerate(tqdm(dlb)):
                optimizer.zero_grad()
                Q = rate_module()
                device = Q.device
                datap = datapoint.to(device=device)
                starting_state, ending_state, branch_length = (
                    datap[:, 0],
                    datap[:, 1],
                    datap[:, 2],
                )
                idx1 = (
                    ending_state.unsqueeze(-1)
                    .repeat(1, num_states)
                    .unsqueeze(-1)
                    .long()
                )
                mats = torch.log(
                    torch.matrix_exp(branch_length[:, None, None] * Q)
                )
                mats = mats.gather(-1, idx1).squeeze(-1)
                mats = mats.gather(
                    1, starting_state.unsqueeze(1).long()
                ).squeeze()

                # loss = -1.0 / m * mats.mean()
                loss = -1.0 / m * mats.sum()
                loss.backward()
                optimizer.step()

                rg.set_description(str(loss.item()), refresh=True)
                if it % n_every == 0:
                    sqrt_dif = (Q - Q_true) * (Q - Q_true)
                    frob_norm = (
                        torch.sqrt(torch.sum(sqrt_dif)).item()
                        if Q_true is not None
                        else 0.0
                    )
                    frob_norm_diag = (
                        torch.sqrt(torch.sum(sqrt_dif.diag())).item()
                        if Q_true is not None
                        else 0.0
                    )
                    frob_norm_offdiag = (
                        torch.sqrt(
                            torch.sum(sqrt_dif - torch.diag(sqrt_dif.diag()))
                        ).item()
                        if Q_true is not None
                        else 0.0
                    )
                    df_res = df_res.append(
                        dict(
                            frob_norm=frob_norm,
                            loss=loss.item(),
                            time=time.time() - start,
                            epoch=epoch,
                            it=it,
                            frob_norm_diag=frob_norm_diag,
                            frob_norm_offdiag=frob_norm_offdiag,
                        ),
                        ignore_index=True,
                    )
                it += 1
                if it >= max_iter:
                    break
    except KeyboardInterrupt:
        return df_res, Q
    return df_res, Q


def train_quantization(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    num_epochs=1000,
    Q_true=None,
    optimizer=None,
):
    """
    Quantization baseline
    """
    if optimizer is None:
        optimizer = torch.optim.SGD(
            rate_module.parameters(), lr=lr, momentum=0.0, weight_decay=0
        )

    print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    try:
        for epoch in rg:
            optimizer.zero_grad()
            Q = rate_module()
            device = Q.device
            # Now compute the loss
            loss = 0.0
            for datapoint in dlb:
                branch_length, cmat = datapoint
                branch_length = branch_length.to(device=device)
                cmat = cmat.to(device=device)

                branch_length_ = branch_length
                mats = torch.log(
                    torch.matrix_exp(branch_length_[:, None, None] * Q)
                )
                mats = mats * cmat
                loss += -1 / m * mats.sum()
            # Take a gradient step.
            loss.backward(retain_graph=True)
            optimizer.step()
            rg.set_description(str(loss.item()), refresh=True)

            if Q_true is not None:
                dif = Q - Q_true
                sqrt_dif = dif * dif
                frob_norm = torch.sqrt(torch.sum(sqrt_dif)).item()
                e, v = torch.eig(dif)
                nuc_norm = e[:, 0].abs().max().item()
                frob_norm_diag = torch.sqrt(torch.sum(sqrt_dif.diag())).item()
                frob_norm_offdiag = torch.sqrt(
                    torch.sum(sqrt_dif - torch.diag(sqrt_dif.diag()))
                ).item()
            else:
                frob_norm = 0.0
                nuc_norm = 0.0
                frob_norm_diag = 0.0
                frob_norm_offdiag = 0.0
            df_res = df_res.append(
                dict(
                    nuc_norm=nuc_norm,
                    frob_norm=frob_norm,
                    loss=loss.item(),
                    time=time.time() - start,
                    epoch=epoch,
                    frob_norm_diag=frob_norm_diag,
                    frob_norm_offdiag=frob_norm_offdiag,
                ),
                ignore_index=True,
            )
    except KeyboardInterrupt:
        return df_res, Q
    return df_res, Q


def train_quantization_N(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    max_iter=20,
    num_epochs=1000,
    Q_true=None,
    optimizer=None,
):
    """
    Quantization baseline
    """
    optimizer = torch.optim.LBFGS(
        rate_module.parameters(), lr=lr, max_iter=max_iter
    )

    print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    for epoch in rg:
        optimizer.zero_grad()
        Q = rate_module()
        device = Q.device

        def closure():
            optimizer.zero_grad()
            loss = 0.0
            for datapoint in dlb:
                branch_length, cmat = datapoint
                branch_length = branch_length.to(device=device)
                cmat = cmat.to(device=device)
                branch_length_ = branch_length
                mats = torch.log(
                    torch.matrix_exp(branch_length_[:, None, None] * Q)
                )
                mats = mats * cmat
                loss += -1 / m * mats.sum()
            loss.backward(retain_graph=True)
            return loss

        # Now compute the loss
        loss = 0.0
        for datapoint in dlb:
            branch_length, cmat = datapoint
            branch_length = branch_length.to(device=device)
            cmat = cmat.to(device=device)

            branch_length_ = branch_length
            mats = torch.log(
                torch.matrix_exp(branch_length_[:, None, None] * Q)
            )
            mats = mats * cmat
            loss += -1 / m * mats.sum()
        # Take a gradient step.
        loss.backward(closure, retain_graph=True)
        optimizer.step()
        rg.set_description(str(loss.item()), refresh=True)

        frob_norm = (
            torch.sqrt(torch.sum((Q - Q_true) * (Q - Q_true))).item()
            if Q_true is not None
            else 0.0
        )
        df_res = df_res.append(
            dict(
                frob_norm=frob_norm,
                loss=loss.item(),
                time=time.time() - start,
                epoch=epoch,
            ),
            ignore_index=True,
        )
    return df_res, Q


def train_diag_param(
    rate_module,
    quantized_dataset,
    m=1.0,
    lr=1.0,
    num_epochs=1000,
    Q_true=None,
    optimizer=None,
):
    num_states = rate_module.num_states

    def construct_X(d, tau):
        # exp_tau_d = torch.expm1(tau * d) + 1.0
        exp_tau_d = torch.exp(tau * d)
        # n_branch_len, n_states
        diag = tau.unsqueeze(-1) * torch.diag_embed(exp_tau_d)

        offdiag = exp_tau_d.unsqueeze(-1) - exp_tau_d.unsqueeze(1)
        offdiag_coeff = d[:, None] - d[None]
        offdiag_coeff += 1e-8 * torch.eye(num_states, device=d.device)
        offdiag_coeff = 1.0 / offdiag_coeff
        offdiag = offdiag * offdiag_coeff

        d_diff = d[:, None] - d[None]
        d_diff += 1e-8 * torch.eye(num_states, device=d.device)
        exp_content = tau.unsqueeze(-1) * d_diff
        common_term = (tau.unsqueeze(-1) * d[None]).exp()
        offdiag = common_term * torch.expm1(exp_content) / d_diff
        offdiag = offdiag * (1.0 - torch.eye(num_states, device=d.device))

        X = diag + offdiag
        return X

    if optimizer is None:
        optimizer = torch.optim.SGD(
            rate_module.parameters(), lr=lr, momentum=0.0, weight_decay=0
        )

    # print(f"Training for {num_epochs} epochs")
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    start = time.time()
    df_res = pd.DataFrame()
    loss = 0.0
    rg = tqdm(range(num_epochs))
    try:
        for epoch in rg:
            optimizer.zero_grad()
            Q = rate_module()
            device = Q.device
            with torch.no_grad():
                v, lambd, uh = torch.linalg.svd(Q, full_matrices=True)
            u = uh.T
            # Now compute the loss
            loss = 0.0
            true_loss = 0.0
            for datapoint in dlb:
                branch_length, cmat = datapoint
                branch_length = branch_length.to(device=device)
                cmat = cmat.to(device=device)

                tau = branch_length[:, None]

                with torch.no_grad():
                    X = construct_X(lambd, tau).float()
                    tmat = torch.matrix_exp(tau.unsqueeze(-1) * Q).float()
                    d = cmat / tmat
                # subloss = (Q * Z).sum()
                subloss = (d * (v @ ((uh @ Q @ v) * X) @ u.T)).sum()
                # subloss = (
                #     d
                #     * (
                #         v @ (
                #             X * (u @ Q @ vh)
                #         )
                #         @ uh
                #     )
                # ).sum()
                loss -= 1.0 / m * subloss
                print("cmat", cmat.min(), cmat.max())
                print("tmat", tmat.min().item(), tmat.max().item())
                # # print("tmat2", tmat2.min().item(), tmat2.max().item())
                print(
                    "tmat log", tmat.log().min().item(), tmat.log().max().item()
                )
                true_loss -= 1.0 / m * (cmat * (tmat.log())).sum()
                # mats = mats * cmat
                # loss += -1 / m * mats.sum()
            # Take a gradient step.
            loss.backward(retain_graph=True)
            optimizer.step()
            rg.set_description(str(loss.item()), refresh=True)

            frob_norm = (
                torch.sqrt(torch.sum((Q - Q_true) * (Q - Q_true))).item()
                if Q_true is not None
                else 0.0
            )
            df_res = df_res.append(
                dict(
                    frob_norm=frob_norm,
                    loss=loss.item(),
                    true_loss=true_loss.item(),
                    time=time.time() - start,
                    epoch=epoch,
                ),
                ignore_index=True,
            )
    except KeyboardInterrupt:
        return df_res, Q
    return df_res, Q


@torch.no_grad()
def estimate_likelihood(
    rate_module,
    quantized_dataset,
    m=1.0,
):
    """
    Quantization baseline
    """
    dlb = DataLoader(quantized_dataset, batch_size=1000, shuffle=False)
    Q = rate_module()
    device = Q.device
    loss = 0.0
    for datapoint in dlb:
        branch_length, cmat = datapoint
        branch_length = branch_length.to(device=device)
        cmat = cmat.to(device=device)

        branch_length_ = branch_length
        mats = torch.log(torch.matrix_exp(branch_length_[:, None, None] * Q))
        mats = mats * cmat
        loss += -1 / m * mats.sum()
    return loss.item()
