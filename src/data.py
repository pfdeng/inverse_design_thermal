"""Dataset loading, feature construction and normalisation.

Input tensor (8 channels):
    0 mask, 1 Xnorm, 2 Ynorm, 3 wall-distance,
    4 Re, 5 Pr, 6 Umean, 7 dp        (scalars broadcast as maps)
Output tensor (4 channels): ux, uy, p, T   (standardised over fluid region)
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _wall_distance(mask):
    # normalised distance to the nearest solid/exterior cell
    d = distance_transform_edt(mask > 0.5).astype(np.float32)
    return d


def load_dataset(path, seed=0, splits=(0.8, 0.1, 0.1), use_dp=True):
    """Load and featurise the dataset.

    use_dp=True  -> 8 input channels (scalars Re,Pr,Umean,dp): the operating-
                    condition surrogate (pressure drop is a given input).
    use_dp=False -> 7 input channels (scalars Re,Pr,Umean): the *forward*
                    operator used for inverse design, where dp is an OUTPUT
                    (read from the predicted pressure field) rather than input.
    """
    d = np.load(path, allow_pickle=True)
    mask = d["mask"].astype(np.float32)          # (N,gy,gx)
    fields = d["fields"].astype(np.float32)      # (N,4,gy,gx)
    scalars = d["scalars"].astype(np.float32)    # (N,6) Re,Pr,Umean,Q,dp,H_in
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    N = mask.shape[0]

    # split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_tr = int(splits[0] * N)
    n_va = int(splits[1] * N)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    # coordinate channels (shared)
    Xn = (X - X.min()) / (X.max() - X.min() + 1e-9)
    Yn = (Y - Y.min()) / (Y.max() - Y.min() + 1e-9)

    # wall distance per sample
    wdist = np.stack([_wall_distance(m) for m in mask])  # (N,gy,gx)

    # ---- normalisation stats from TRAIN split ----------------------------
    # scalar inputs used: Re(0), Pr(1), Umean(2), and optionally dp(4)
    sc_cols = [0, 1, 2, 4] if use_dp else [0, 1, 2]
    n_scalar = len(sc_cols)
    in_ch = 4 + n_scalar          # mask, Xn, Yn, wall-dist, + scalars
    sc = scalars[:, sc_cols]
    sc_mean = sc[tr].mean(0)
    sc_std = sc[tr].std(0) + 1e-6
    wd_mean = wdist[tr].mean()
    wd_std = wdist[tr].std() + 1e-6

    # output stats over fluid region of train split
    m_tr = mask[tr][:, None]                     # (ntr,1,gy,gx)
    fld_tr = fields[tr]
    fmean = np.zeros(4, np.float32)
    fstd = np.zeros(4, np.float32)
    for k in range(4):
        vals = fld_tr[:, k][m_tr[:, 0] > 0.5]
        fmean[k] = vals.mean()
        fstd[k] = vals.std() + 1e-6

    gy, gx = mask.shape[1], mask.shape[2]

    def build(index):
        n = index.size
        inp = np.zeros((n, in_ch, gy, gx), np.float32)
        out = np.zeros((n, 4, gy, gx), np.float32)
        nu = np.zeros(n, np.float32)
        for j, i in enumerate(index):
            inp[j, 0] = mask[i]
            inp[j, 1] = Xn
            inp[j, 2] = Yn
            inp[j, 3] = (wdist[i] - wd_mean) / wd_std * (mask[i] > 0.5)
            scn = (scalars[i, sc_cols] - sc_mean) / sc_std
            for c in range(n_scalar):
                inp[j, 4 + c] = scn[c]
            for k in range(4):
                out[j, k] = (fields[i, k] - fmean[k]) / fstd[k] * (mask[i] > 0.5)
            Re, Umean, H_in = scalars[i, 0], scalars[i, 2], scalars[i, 5]
            nu[j] = Umean * H_in / Re
        return (torch.from_numpy(inp), torch.from_numpy(out),
                torch.from_numpy(mask[index]), torch.from_numpy(nu))

    stats = {"fmean": fmean, "fstd": fstd, "sc_mean": sc_mean, "sc_std": sc_std,
             "wd_mean": wd_mean, "wd_std": wd_std, "sc_cols": sc_cols,
             "use_dp": use_dp, "in_ch": in_ch}
    data = {"train": build(tr), "val": build(va), "test": build(te),
            "test_idx": te, "stats": stats,
            "raw": {"mask": mask, "fields": fields, "scalars": scalars,
                    "X": X, "Y": Y}}
    return data
