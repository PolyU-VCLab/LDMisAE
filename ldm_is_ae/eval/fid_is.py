#!/usr/bin/env python3
# FID / Inception-Score evaluator in the ADM (evaluator_adm, TF) caliber.
#   FID: Inception-V3 pool2048 features compared against a reference stats .npz holding mu / sigma.
#   IS:  softmax(pool2048 @ w) with w = fc.weight.T (bit-identical to the TF graph weights),
#        sequential 10 splits with the improved-gan formula -- this is what evaluator_adm's
#        compute_inception_score does. torch_fidelity's ISC differs because it shuffles
#        (RandomState(2020)) before splitting instead of splitting sequentially.
# Usage: python3 fid_is.py --sample_path <sample_dir> --ref_stats <stats.npz> [--tag name] [--out_json out.json]
import argparse, glob, json, os, time
import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.datasets import ImagesPathDataset

BATCH = 50
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def build_fe(weights_path=None):
    fe = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"],
                                     feature_extractor_weights_path=weights_path)
    return fe.to(DEV).eval()

def extract(fe, d):
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    print("   files:", len(files), flush=True)
    dl = DataLoader(ImagesPathDataset(files), batch_size=BATCH, num_workers=8, shuffle=False)
    F, G = [], []
    with torch.no_grad():
        for i, x in enumerate(dl):
            x = x.to(DEV)
            f, lg = fe(x)
            F.append(f.double().cpu().numpy()); G.append(lg.double().cpu().numpy())
            if i % 100 == 0: print("   batch %d" % i, flush=True)
    return np.concatenate(F, 0), np.concatenate(G, 0)

def fid_between(m1, s1, m2, s2):
    d = m1 - m2
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(cm): cm = cm.real
    return float(d.dot(d) + np.trace(s1 + s2 - 2.0 * cm))

def softmax_np(lg):
    x = lg - lg.max(1, keepdims=True)
    p = np.exp(x); p /= p.sum(1, keepdims=True)
    return p

def inception_score_seq(p, splits=10):
    # evaluator_adm style: sequential splits (split_size=N//splits), improved-gan formula
    N = p.shape[0]
    split_size = N // splits
    scores = []
    for i in range(0, N, split_size):
        part = p[i:i + split_size]
        py = np.mean(part, 0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        scores.append(float(np.exp(np.mean(np.sum(kl, 1)))))
    return float(np.mean(scores)), float(np.std(scores))

def inception_score_shuffled(p, splits=10, seed=2020):
    # torch_fidelity style: shuffle before splitting (kept for cross-check)
    N = p.shape[0]; idx = np.arange(N); np.random.RandomState(seed).shuffle(idx)
    scores = []
    for part in np.array_split(idx, splits):
        pp = p[part]
        py = pp.mean(0, keepdims=True)
        kl = pp * (np.log(pp + 1e-12) - np.log(py + 1e-12))
        scores.append(float(np.exp(kl.sum(1).mean())))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Command-line front end.  The numerics above are untouched
# (fid_between / softmax_np / inception_score_seq / inception_score_shuffled).
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="FID + Inception Score in the ADM / evaluator_adm (TF) caliber, torch implementation.")
    p.add_argument("--sample_path", help="directory holding the generated *.png samples")
    p.add_argument("--ref_stats", help="reference .npz with 'mu' and 'sigma' (e.g. VIRTUAL_imagenet256_labeled.npz)")
    p.add_argument("--tag", default=None, help="name of this run (default: basename of --sample_path)")
    p.add_argument("--out_json", default=None,
                   help="output json (default: <parent of --sample_path>/<tag>_fid_is.json)")
    p.add_argument("--weights", default=None,
                   help="Inception-V3 weights .pth (default: torch-fidelity cache / auto download)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return p


def main_cli(argv=None):
    global BATCH, DEV
    args = build_argparser().parse_args(argv)
    if not args.sample_path or not args.ref_stats:
        build_argparser().error("--sample_path and --ref_stats are required")
    if args.device:
        DEV = args.device
    if args.batch_size:
        BATCH = args.batch_size
    # Inception-V3 weights: use --weights when given, otherwise let torch-fidelity resolve
    # them from its own cache (downloading on first use).
    tag = args.tag or os.path.basename(os.path.abspath(args.sample_path))
    outj = args.out_json or os.path.join(os.path.dirname(os.path.abspath(args.sample_path)),
                                        tag + "_fid_is.json")
    z = np.load(args.ref_stats, allow_pickle=True)
    ref_mu = np.asarray(z["mu"], dtype=np.float64)
    ref_sigma = np.asarray(z["sigma"], dtype=np.float64)
    fe = build_fe(args.weights)
    t0 = time.time()
    f, g = extract(fe, args.sample_path)
    mu = f.mean(0); sigma = np.cov(f, rowvar=False)
    fid = fid_between(mu, sigma, ref_mu, ref_sigma)
    p = softmax_np(g)
    is_mean, is_std = inception_score_seq(p)
    is_shuf, is_shuf_std = inception_score_shuffled(p)
    res = {tag: {"fid": fid, "is_mean": is_mean, "is_std": is_std,
                 "is_mean_shuffled": is_shuf, "is_std_shuffled": is_shuf_std,
                 "n": int(f.shape[0]), "secs": round(time.time() - t0, 1)}}
    print("DONE", tag, res[tag], flush=True)
    json.dump(res, open(outj, "w"), indent=2)
    print("wrote", outj, flush=True)
    return res


if __name__ == "__main__":
    main_cli()
