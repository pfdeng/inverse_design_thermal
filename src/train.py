"""Train the U-Net thermal-flow surrogate (data loss + physics-informed loss)."""
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_dataset
from .model import UNet
from .physics import physics_loss


def masked_mse(pred, target, mask):
    m = mask.unsqueeze(1)
    diff = (pred - target) ** 2 * m
    return diff.sum() / (m.sum() * pred.shape[1] + 1e-6)


def denorm(pred, fmean, fstd, mask):
    phys = pred * fstd.view(1, -1, 1, 1) + fmean.view(1, -1, 1, 1)
    return phys * mask.unsqueeze(1)


def iterate_batches(tensors, bs, device, shuffle=True):
    inp, out, mask, nu = tensors
    n = inp.shape[0]
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        yield (inp[idx].to(device), out[idx].to(device),
               mask[idx].to(device), nu[idx].to(device))


def evaluate(model, tensors, device, fmean, fstd):
    model.eval()
    tot, ntot = 0.0, 0
    with torch.no_grad():
        for inp, out, mask, nu in iterate_batches(tensors, 8, device, shuffle=False):
            pred = model(inp)
            tot += masked_mse(pred, out, mask).item() * inp.shape[0]
            ntot += inp.shape[0]
    return tot / ntot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--lam_div", type=float, default=2e-2)
    ap.add_argument("--lam_mom", type=float, default=2e-4)
    ap.add_argument("--phys_warmup", type=int, default=30)
    ap.add_argument("--out", default="results/model.pt")
    ap.add_argument("--no_dp", action="store_true",
                    help="drop the dp input channel -> forward operator for inverse design")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)

    data = load_dataset(args.data, use_dp=not args.no_dp)
    stats = data["stats"]
    in_ch = stats["in_ch"]
    fmean = torch.tensor(stats["fmean"], device=device)
    fstd = torch.tensor(stats["fstd"], device=device)

    model = UNet(in_ch=in_ch, out_ch=4, base=args.base).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e6:.2f}M | train {data['train'][0].shape[0]} "
          f"val {data['val'][0].shape[0]} test {data['test'][0].shape[0]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val = np.inf
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        phys_w = min(1.0, max(0.0, (ep - args.phys_warmup) / 30.0)) if ep >= args.phys_warmup else 0.0
        agg = {"data": 0.0, "div": 0.0, "mom": 0.0}
        nb = 0
        for inp, out, mask, nu in iterate_batches(data["train"], args.bs, device):
            pred = model(inp)
            l_data = masked_mse(pred, out, mask)
            phys = denorm(pred, fmean, fstd, mask)
            l_div, l_mom = physics_loss(phys, nu, mask)
            loss = l_data + phys_w * (args.lam_div * l_div + args.lam_mom * l_mom)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            agg["data"] += l_data.item(); agg["div"] += l_div.item(); agg["mom"] += l_mom.item()
            nb += 1
        sched.step()
        val = evaluate(model, data["val"], device, fmean, fstd)
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "stats": stats,
                        "args": vars(args)}, args.out)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} | data {agg['data']/nb:.4f} div {agg['div']/nb:.3e} "
                  f"mom {agg['mom']/nb:.3e} | val {val:.4f} | best {best_val:.4f} "
                  f"| phys_w {phys_w:.2f} | {time.time()-t0:.0f}s")

    print(f"done. best val {best_val:.4f}. saved -> {args.out}")


if __name__ == "__main__":
    main()
