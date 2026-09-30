"""Standalone single-GPU generation / online-eval style inference.

Single GPU:
    python3 inference.py --model JiT-B/half --num_images 10000 --cfg 1.0 --checkpoint ... --output_dir ...

Multi GPU (race on batches via --random_idx + file-existence skip):
    CUDA_VISIBLE_DEVICES=0 python3 inference.py ... --num_images 10000 --random_idx &
    CUDA_VISIBLE_DEVICES=1 python3 inference.py ... --num_images 10000 --random_idx &
    ...                                       ...                           ...
    CUDA_VISIBLE_DEVICES=7 python3 inference.py ... --num_images 10000 --random_idx &
    wait
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ldm_is_ae.denoiser import Denoiser, load_denoiser_checkpoint_model, unwrap_torch_compile_net_keys
from ldm_is_ae.utils.train_log import configure_train_logger, logger


def _eval_generate_ctx(args, device):
    if getattr(args, "sample_bf16", False) and device.type == "cuda" and not getattr(args, "fp32", False):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _eval_swap_params_to_ema1(model_without_ddp):
    ema_list = getattr(model_without_ddp, "ema_params1", None)
    if ema_list is None:
        raise RuntimeError("ema_params1 is missing; load a checkpoint with model_ema1")
    named = list(model_without_ddp.named_parameters())
    if len(ema_list) != len(named):
        raise RuntimeError("ema_params1 len {} != named_parameters {}".format(len(ema_list), len(named)))
    train_backup = [p.detach().clone() for (_, p) in named]
    for (name, p), ema in zip(named, ema_list):
        e = ema.to(device=p.device, dtype=p.dtype, non_blocking=True)
        if e.shape != p.shape:
            raise RuntimeError("EMA vs param shape mismatch {}: {} {}".format(name, tuple(e.shape), tuple(p.shape)))
        p.data.copy_(e)
    return train_backup


def _eval_restore_train_params(model_without_ddp, train_backup):
    for (_, p), b in zip(model_without_ddp.named_parameters(), train_backup):
        p.data.copy_(b.to(device=p.device, dtype=p.dtype, non_blocking=True))


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    gen_bs = int(batch_size)
    num_groups = (args.num_images + gen_bs - 1) // gen_bs

    save_folder = getattr(args, "gen_images_dir", None) or os.path.join(
        args.output_dir,
        "epoch{:04d}-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            epoch, model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    logger.info("Save to: {}", save_folder)
    os.makedirs(save_folder, exist_ok=True)

    ema_mode = getattr(args, "ema_mode", "ema1")
    if ema_mode != "none":
        logger.info("Switch to ema")
        train_backup = _eval_swap_params_to_ema1(model_without_ddp)
    else:
        train_backup = None
    mod_dev = next(model_without_ddp.parameters()).device

    assert args.num_images > 0 and args.class_num > 0
    total_slots = num_groups * gen_bs
    class_label_gen_world = np.zeros(total_slots, dtype=np.int64)
    class_label_gen_world[:args.num_images] = (np.arange(args.num_images, dtype=np.int64) % int(args.class_num))

    base_start_idx = np.arange(num_groups, dtype=np.int64) * gen_bs
    if getattr(args, "random_idx", False):
        ephem = int.from_bytes(os.urandom(8), "little")
        rng = np.random.default_rng(ephem)
        all_start_idx = base_start_idx[rng.permutation(num_groups)]
        logger.info("random_idx: shuffle; ephemeral_shuffle_seed={}", ephem)
    else:
        all_start_idx = base_start_idx
    start_idx_list = list(all_start_idx)

    for step_i, start_idx in enumerate(start_idx_list):
        logger.info("Generation step {}/{}", step_i + 1, len(start_idx_list))
        g = start_idx // gen_bs
        end_idx = start_idx + gen_bs
        lo, hi = int(start_idx), min(int(end_idx), int(args.num_images))
        batch_done = lo >= hi or all(
            os.path.isfile(os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5)))) for img_id in range(lo, hi)
        )
        if batch_done:
            logger.info(
                "skip: step={}/{} start_idx={} img_id=[{}, {}) {}",
                step_i + 1, len(start_idx_list), start_idx, lo, hi,
                "(no images)" if lo >= hi else "(all png exist)",
            )
            continue
        labels_gen = torch.as_tensor(class_label_gen_world[start_idx:end_idx], dtype=torch.long, device=mod_dev).contiguous()
        global_seed = int(getattr(args, "seed", 0))
        gen_seed = int(start_idx + global_seed)
        torch.manual_seed(gen_seed)
        torch.cuda.manual_seed_all(gen_seed)
        with _eval_generate_ctx(args, mod_dev):
            sampled_images = model_without_ddp.generate(labels_gen)
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()
        for b_id in range(sampled_images.size(0)):
            img_id = g * gen_bs + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    if train_backup is not None:
        logger.info("Switch back from ema")
        _eval_restore_train_params(model_without_ddp, train_backup)


def get_inference_parser():
    p = argparse.ArgumentParser("JiT inference", add_help=True)
    p.add_argument("--resume", default="", type=str, help="Dir containing checkpoint-last.pth (same as training --resume)")
    p.add_argument("--checkpoint", default="", type=str, help="Explicit .pth path; overrides checkpoint-last under --resume")
    p.add_argument("--init_jit_ckpt", default="", type=str, help="Path to a non-half JiT checkpoint (.pth). Wraps it into /half Denoiser via net.model_dec.*; model_enc.* stays fresh. Requires --model to be a /half variant; ignored if --checkpoint/--resume is set.")
    p.add_argument("--output_dir", default="./output_dir", type=str, help="Base dir; a subdir cfg*_imin*_imax*_s* is appended after ckpt merge; args.json and imgs/ live there")
    p.add_argument("--epoch", default=0, type=int, help="Epoch tag for save folder name (online_eval uses training epoch)")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--compute_metrics", action="store_true", help="Run FID/IS via torch_fidelity and log scalars to TensorBoard")
    p.add_argument("--lpips_weight", default=0.0, type=float,
                   help="Weight of the LPIPS term. The sampling path never needs it, so the default 0.0 builds no LPIPS branch and no vgg.pth is required")
    p.add_argument("--lpips_model_path", default="", type=str,
                   help="Path to the LPIPS/VGG weights; used when --lpips_weight != 0 and overrides the value stored in the checkpoint")
    # model (match the Denoiser)
    p.add_argument("--model", default="JiT-B/16", type=str)
    p.add_argument("--img_size", default=256, type=int)
    p.add_argument("--zc", type=int, default=3)
    p.add_argument("--placeholder_k", default=None, type=float,
                   help="Eq.4 mixing exponent k in gamma(t)=t^k; None = inherit from checkpoint args (paper: 3); "
                        "checkpoints without this key fall back to 0.0 (pre-flag behaviour)")
    p.add_argument("--pixel_patch_size", type=int, default=16)
    p.add_argument("--depth_enc", default=None, type=int)
    p.add_argument("--depth_dec", default=None, type=int)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--proj_dropout", type=float, default=0.0)
    p.add_argument("--class_num", default=1000, type=int)
    p.add_argument("--label_drop_prob", type=float, default=0.1)
    p.add_argument("--P_mean", default=-0.8, type=float)
    p.add_argument("--P_std", default=0.8, type=float)
    p.add_argument("--noise_scale", default=1.0, type=float)
    p.add_argument("--t_eps", default=5e-2, type=float)
    p.add_argument("--ema_decay1", type=float, default=0.9999)
    p.add_argument("--ema_decay2", type=float, default=0.9996)
    # sampling (match evaluate / Denoiser)
    p.add_argument("--sampling_method", default="heun", type=str)
    p.add_argument("--num_sampling_steps", default=50, type=int)
    p.add_argument("--cfg", default=1.0, type=float)
    p.add_argument("--interval_min", default=0.0, type=float)
    p.add_argument("--interval_max", default=1.0, type=float)
    p.add_argument("--deltat", default=0.0, type=float, help="Shift t by -deltat when feeding to model")
    p.add_argument("--schedule_power", default=1.0, type=float, help=">1 = more steps near clean end (power schedule); 1.0 = linear")
    p.add_argument("--first_step_pr", action="store_true", help="Predict-and-renoise at step 0 (backward compat; use --pr_steps for N-step PR)")
    p.add_argument("--pr_steps", default=0, type=int, choices=[0, 1, 2, 3], help="Apply PR for first N steps (0=off, 1=single-step PR, 2/3=multi-step)")
    p.add_argument("--xp_ste", action="store_true", help="Apply 8-bit STE quantization on xp in forward_ldm")
    p.add_argument("--ema_mode", default="ema1", choices=["none", "ema1", "ema2"],
                    help="EMA weight mode for evaluation: none (training weights), ema1, ema2 (default: ema1)")
    p.add_argument("--num_images", default=50000, type=int, help="Total images; must be divisible by min(num_images, class_num)")
    p.add_argument("--gen_bsz", type=int, default=256, help="Generation batch size")
    p.add_argument("--random_idx", action="store_true", help="Shuffle start_idx order before generation")
    return p


CLI_INFERENCE_KEYS = (
    "resume", "checkpoint", "output_dir", "epoch", "seed", "compute_metrics", "gen_bsz", "num_images",
    "num_sampling_steps", "cfg", "interval_min", "interval_max", "sampling_method", "device",
    "noise_scale", "t_eps",
    "deltat", "schedule_power", "first_step_pr", "pr_steps", "xp_ste", "ema_mode",
    "lpips_weight", "lpips_model_path",
)


def _ckpt_arg(checkpoint, name):
    """Read one training-time hyper-parameter from checkpoint['args']; None when missing."""
    ca = checkpoint.get("args")
    if isinstance(ca, argparse.Namespace):
        return vars(ca).get(name)
    if isinstance(ca, dict):
        return ca.get(name)
    return None


def _resolve_placeholder_k(args, cli_k, ckpt_k):
    """Eq.4 mixing-exponent precedence: explicit CLI > value stored in the checkpoint > 0.0 (older checkpoints have no such key, i.e. no mixing)."""
    if cli_k is not None:
        if ckpt_k is not None and float(ckpt_k) != float(cli_k):
            logger.warning("placeholder_k: CLI {} differs from checkpoint training value {} -> sampling uses a "
                           "different Eq.4 mixing exponent than training", cli_k, ckpt_k)
        args.placeholder_k = float(cli_k)
    elif ckpt_k is not None:
        args.placeholder_k = float(ckpt_k)
    else:
        args.placeholder_k = 0.0
        logger.info("placeholder_k: not found in checkpoint args -> using 0.0 (no Eq.4 mixing; pre-flag code)")


def _merge_ckpt_training_args(args, checkpoint):
    ca = checkpoint.get("args")
    if ca is None:
        return
    if isinstance(ca, argparse.Namespace):
        items = vars(ca).items()
    elif isinstance(ca, dict):
        items = ca.items()
    else:
        return
    skip = {"gpu", "rank", "distributed", "dist_backend", "random_idx"}
    for k, v in items:
        if k.startswith("_") or k in skip or isinstance(v, torch.Tensor):
            continue
        setattr(args, k, v)


def _jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


def _resolve_checkpoint_path(args):
    """Return (path, is_init_jit). Precedence: --checkpoint > --resume (when the ckpt exists) > --init_jit_ckpt.
    If the --resume directory holds no checkpoint-last.pth it silently falls through to --init_jit_ckpt,
    matching the behaviour of the training entry point."""
    if args.checkpoint:
        cp = args.checkpoint
        if not os.path.isfile(cp):
            raise FileNotFoundError(cp)
        return cp, False
    if args.resume:
        cp = os.path.join(args.resume, "checkpoint-last.pth")
        if os.path.isfile(cp):
            return cp, False
        logger.info("_resolve_checkpoint_path: --resume '{}' has no checkpoint-last.pth; trying --init_jit_ckpt", args.resume)
    if getattr(args, "init_jit_ckpt", ""):
        cp = args.init_jit_ckpt
        if not os.path.isfile(cp):
            raise FileNotFoundError(cp)
        assert "/half" in args.model, f"--init_jit_ckpt requires a /half --model (e.g. JiT-H/half), got {args.model}"
        return cp, True
    raise ValueError("Provide --checkpoint path, --resume dir (with checkpoint-last.pth), or --init_jit_ckpt path")


def _infer_sampling_tag(args):
    def _r(x):
        return str(float(x)).replace(".", "p").replace("-", "m")
    dt = float(getattr(args, "deltat", 0.0))
    dt_tag = f"_dt{_r(dt)}" if dt != 0.0 else ""
    sp = float(getattr(args, "schedule_power", 1.0))
    sp_tag = f"_sp{_r(sp)}" if sp != 1.0 else ""
    pr = int(getattr(args, "pr_steps", 0))
    if pr == 0 and bool(getattr(args, "first_step_pr", False)):
        pr = 1
    pr_tag = f"_pr{pr}" if pr > 0 else ""
    xp_tag = "_xpste" if getattr(args, "xp_ste", False) else ""
    em = getattr(args, "ema_mode", "ema1")
    em_tag = {"none": "_emanone", "ema1": "", "ema2": "_ema2"}.get(em, "")
    return "cfg{}_imin{}_imax{}_s{}{}{}{}{}{}".format(_r(args.cfg), _r(args.interval_min), _r(args.interval_max), int(args.num_sampling_steps), dt_tag, sp_tag, pr_tag, xp_tag, em_tag)


def main():
    args = get_inference_parser().parse_args()
    snap = {k: getattr(args, k) for k in CLI_INFERENCE_KEYS if hasattr(args, k)}
    cli_k = getattr(args, "placeholder_k", None)   # None = not given on the command line; sampling then follows the checkpoint

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("inference/evaluate path requires CUDA")
    args.gpu = 0
    args.distributed = False
    device = torch.device("cuda", args.gpu)
    torch.set_float32_matmul_precision('high')

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.num_images <= 0:
        raise ValueError("num_images must be positive")
    if args.class_num <= 0:
        raise ValueError("class_num must be positive")
    eff = min(args.num_images, args.class_num)
    if args.num_images % eff != 0:
        raise ValueError("num_images must be divisible by min(num_images, class_num)")

    ckpt_path, is_init_jit = _resolve_checkpoint_path(args)
    if is_init_jit:
        logger.info("Using --init_jit_ckpt path; CLI args (not ckpt args) will be used for /half Denoiser config.")
        _resolve_placeholder_k(args, cli_k, None)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_k = _ckpt_arg(checkpoint, "placeholder_k")
        _merge_ckpt_training_args(args, checkpoint)
        for k, v in snap.items():
            setattr(args, k, v)
        _resolve_placeholder_k(args, cli_k, ckpt_k)

    args.output_dir = os.path.join(os.path.abspath(args.output_dir), _infer_sampling_tag(args))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    configure_train_logger(args.output_dir)
    logger.info("Job: {}", Path(__file__).resolve())
    logger.opt(raw=True).info("Arguments:\n" + str(args).replace(", ", ",\n"))
    log_writer = None
    if args.compute_metrics:
        from torch.utils.tensorboard import SummaryWriter
        log_writer = SummaryWriter(log_dir=args.output_dir)

    model_without_ddp = Denoiser(args)
    model_without_ddp.to(device)
    if is_init_jit:
        from ldm_is_ae.train import load_jit_to_half_state
        load_jit_to_half_state(ckpt_path, model_without_ddp, None, device)
    else:
        # Released checkpoints are ema1-only (no 'model' / 'model_ema2'): fall back to
        # model_ema1 as the base state so the default (--ema_mode ema1) path still loads.
        main_sd = checkpoint.get("model")
        if main_sd is None:
            if args.ema_mode == "none":
                raise ValueError("--ema_mode none requires checkpoint['model']; this checkpoint is ema1-only")
            main_sd = checkpoint["model_ema1"]
            logger.info("checkpoint has no 'model' key; using model_ema1 weights as the base state")
        load_denoiser_checkpoint_model(model_without_ddp, main_sd)
        if args.ema_mode != "none":
            ema_state_dict1 = unwrap_torch_compile_net_keys(checkpoint["model_ema1"])
            model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
            ema2_sd = checkpoint.get("model_ema2")
            if ema2_sd is None:
                if args.ema_mode == "ema2":
                    raise ValueError("--ema_mode ema2 requires checkpoint['model_ema2']; this checkpoint is ema1-only")
                logger.info("checkpoint has no 'model_ema2' key; ema2 slots left unset (sampling uses ema1)")
            else:
                ema_state_dict2 = unwrap_torch_compile_net_keys(ema2_sd)
                model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        del checkpoint
    logger.info("Loaded checkpoint from {}", ckpt_path)

    args.gen_images_dir = os.path.join(args.output_dir, "imgs")
    os.makedirs(args.gen_images_dir, exist_ok=True)
    dump = {k: _jsonable(v) for k, v in vars(args).items() if not str(k).startswith("_")}
    dump["checkpoint_path"] = os.path.abspath(ckpt_path)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, ensure_ascii=False)

    ep = int(getattr(args, "epoch", 0))
    with torch.no_grad():
        evaluate(model_without_ddp, args, ep, batch_size=args.gen_bsz, log_writer=log_writer)
    if log_writer is not None:
        log_writer.flush()
    logger.info("Inference done.")


if __name__ == "__main__":
    main()
