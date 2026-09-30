import os
# Before numpy: MKL INTEL threading conflicts with libgomp (e.g. PyTorch); GNU matches OpenMP stack
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import argparse
import datetime
import functools
import numpy as np
import time
from pathlib import Path

import torch
import torch.multiprocessing as torch_mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ldm_is_ae.utils.crop import center_crop_arr
import ldm_is_ae.utils.misc as misc

import copy
from ldm_is_ae.engine import train_one_epoch, evaluate

from ldm_is_ae.denoiser import Denoiser, load_denoiser_checkpoint_model, unwrap_torch_compile_net_keys, jit_checkpoint_model_sd
from ldm_is_ae.utils.train_log import configure_train_logger, logger


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--zc', type=int, default=3, help='Denoiser / net patch-grid channels (B,zc,H//p,W//p); use 3 or 3*p^2 with --pixel_patch_size p')
    parser.add_argument('--pixel_patch_size', type=int, default=16, metavar='P', help='Patch stride p; img_size must be divisible by p (match JiT-B/16 vs /32)')
    parser.add_argument('--depth_enc', default=None, type=int, help='JiT-*/half only: encoder depth (default: B=6 L=12 H=16); must be > in_context_start (B>4 L>8 H>10)')
    parser.add_argument('--depth_dec', default=None, type=int, help='JiT-*/half only: decoder depth (default: same as depth_enc per variant); must be > in_context_start like depth_enc')
    parser.add_argument('--lpips_weight', default=0.1, type=float, help='Weight for LPIPS on full-res RGB (0 disables LPIPS branch)')
    parser.add_argument('--lpips_model_path', default='', type=str, help='Path to LPIPS/VGG weights; empty = auto search under repo (taming paths)')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch = batch_size * world_size * accum_iter)')
    parser.add_argument('--accum_iter', type=int, default=1,
                        help='Gradient accumulation steps (optimizer step every accum_iter micro-batches)')
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                        help='torch.nn.utils.clip_grad_norm_ max norm; 0 disables (applied at each accum boundary before optimizer.step)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--enc_lr_scale', type=float, default=1.0,
                        help='JiT-*/half: LR multiplier for net.model_enc.* vs the decoder (param_group lr_scale; in util.lr_sched lr *= lr_scale). The default 1.0 still uses half grouping.')
    parser.add_argument('--enc_lr_scale_decay', type=float, default=1.0,
                        help='Decay factor applied to enc_lr_scale every enc_lr_scale_decay_epochs epochs; 1.0 = no decay')
    parser.add_argument('--enc_lr_scale_decay_epochs', type=int, default=50,
                        help='Number of epochs between enc_lr_scale decays')
    parser.add_argument('--dec_lr_scale', type=float, default=1.0,
                        help='JiT-*/half: LR multiplier for decoder params (vs encoder lr_scale). Default 1.0 = no change.')
    parser.add_argument('--dec_lr_scale_decay', type=float, default=1.0,
                        help='Decay factor applied to dec_lr_scale every dec_lr_scale_decay_epochs epochs; 1.0 = no decay')
    parser.add_argument('--dec_lr_scale_decay_epochs', type=int, default=200,
                        help='Number of epochs between dec_lr_scale decays')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--v_mse_weight', default=2.0, type=float,
                        help='Weight for velocity MSE loss (LDM branch); 2.0 = legacy hardcoded coefficient')
    parser.add_argument('--xp_ste', action='store_true',
                        help='Apply 8-bit straight-through estimator on xp before encoder')
    parser.add_argument('--enc_bp', default=1, type=int,
                        help='Encoder input->output direct bypass (z = raw + linear_proj(xs)); 0 disables it (z = raw)')
    parser.add_argument('--z_norm', default=None, type=str,
                        help='JiT-*/half: normalize model_enc output latent (rms = RMSNormNoScale, no learnable scale)')
    parser.add_argument('--placeholder_k', default=3.0, type=float,
                        help='Eq.4 time-aware auxiliary feature mixing exponent k in gamma(t)=t^k. '
                             'Paper setting is 3; 0 disables mixing (encoder input = xs, pre-flag behaviour)')

    # end_epoch: freeze enc_lr_scale to 0 after this epoch (0=disable)
    parser.add_argument('--enc_lr_scale_end_epoch', default=0, type=int,
                        help='Epoch after which enc_lr_scale is forced to 0 (freeze encoder). 0=disable.')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=5000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--random_idx', action='store_true',
                        help='Shuffle each rank start_idx order before generation')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--text_dim', default=0, type=int,
                        help='Text embedding dimension (2048 for Qwen3); 0 = class-conditional')
    parser.add_argument('--text_len', default=128, type=int,
                        help='Number of text tokens per prompt (Qwen3 default=128)')
    parser.add_argument('--blip3o_path', default='', type=str,
                        help='Path to BLIP3o-Pretrain-Long-Caption tar directory')
    parser.add_argument('--blip3o_60k_path', default='', type=str,
                        help='Path to BLIP3o-60k SFT tar directory (takes precedence over --blip3o_path)')
    parser.add_argument('--text_encoder_path', default='', type=str,
                        help='Path to Qwen3-1.7B weights (for T2I text encoding)')
    parser.add_argument('--eval_prompts', type=str, default='',
                        help='Path to eval prompts file (one caption per line, for T2I)')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--init_jit_ckpt', default='', type=str,
                        help='Path to a non-half JiT checkpoint (.pth). Loaded into /half net.model_dec.*; net.model_enc.* stays fresh. Only used when --resume ckpt is absent.')
    parser.add_argument('--init_from_ckpt', default='', type=str,
                        help='Path to a checkpoint (.pth) used to PARTIALLY initialize the model (shape-matched keys only; no optimizer/epoch). T2I warm-start from the 512 class-conditional model.')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--fp32', action='store_true',
                        help='Train/eval without CUDA bf16 autocast (fp32 forward)')
    parser.add_argument('--tryrun', action='store_true',
                        help='Shrink per-GPU batch, run one optimizer step, save checkpoint-last, online eval, exit')
    parser.add_argument('--tryrun_div', type=int, default=2,
                        help='tryrun: batch_size = max(1, batch_size // tryrun_div)')
    parser.add_argument('--tryrun_num_images', type=int, default=10,
                        help='tryrun eval: cap num_images, floored to multiple of min(cap, class_num) (0 = use --num_images)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='S1: freeze net.model_{dec,enc} backbone, keep T2I text branch (text_pos_embed / y_embedder.*) trainable')
    parser.add_argument('--max_total_optimizer_steps', default=0, type=int,
                        help='If >0, stop training once cumulative optimizer steps reach this value (step-based length; 0=disable)')
    parser.add_argument('--save_every_n_steps', type=int, default=0,
                        help='Save checkpoint every N optimizer steps (0 = epoch-based only)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run online_eval every N optimizer steps (0 = epoch-based; requires --online_eval)')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def freeze_backbone(denoiser):
    """S1: freeze the net.model_dec.* / net.model_enc.* backbone and keep only the newly added T2I text branch trainable.
    Criterion: the parameter name contains text_pos_embed or y_embedder."""
    frozen, kept = 0, 0
    for name, p in denoiser.named_parameters():
        if not (name.startswith('net.model_dec.') or name.startswith('net.model_enc.')):
            continue
        if ('text_pos_embed' in name) or ('y_embedder' in name):
            kept += p.numel()
            continue
        p.requires_grad_(False)
        frozen += p.numel()
    logger.info("freeze_backbone: frozen {:.6f}M params; kept trainable text branch {:.6f}M", frozen / 1e6, kept / 1e6)
    return frozen, kept


def load_init_from_ckpt(ckpt_path, denoiser):
    """Partial init from a checkpoint: load only the model.* keys whose shapes match (the new T2I text
    layers stay randomly initialised); optimizer / EMA / epoch are not restored."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    sd = unwrap_torch_compile_net_keys(jit_checkpoint_model_sd(sd, denoiser))
    cur = denoiser.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in cur and hasattr(v, "shape") and tuple(cur[k].shape) == tuple(v.shape):
            cur[k] = v.to(dtype=cur[k].dtype)
            loaded.append(k)
        else:
            skipped.append(k)
    missing = [k for k in cur if k not in sd]
    denoiser.load_state_dict(cur, strict=True)
    logger.info("init_from_ckpt: loaded {}/{} keys; skipped {}; fresh(missing) {}", len(loaded), len(sd), len(skipped), len(missing))
    logger.info("init_from_ckpt: fresh keys sample = {}", sorted(missing)[:8])
    logger.info("init_from_ckpt: skipped keys sample = {}", sorted(skipped)[:8])
    del ckpt
    return len(loaded), missing


def load_jit_to_half_state(ckpt_path, denoiser, optimizer, device):
    """Load a non-half JiT checkpoint (e.g. JiT-H/16) into the /half-wrapped Denoiser.
    Mapping rules (aligned by parameter / buffer name, migrated only on an exact shape match;
    mismatches and newly added entries keep the current init):
      - state_dict: old net.* -> new net.model_dec.*; all other prefixes load as-is.
        When zc == xc, final_layer.linear has the same shape and migrates automatically; when zc != xc
        it is skipped and stays zero-initialised. The new net.model_enc.* / bypass are absent from old
        checkpoints and keep the current init.
      - EMA: rebuilt by the same name rules; missing / mismatched entries fall back to the current init.
      - optimizer: Adam moments (exp_avg / exp_avg_sq / step) migrate by parameter name; new parameters
        keep a fresh state.
    Returns: epoch+1 from the checkpoint (for args.start_epoch), or None when unavailable.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert getattr(denoiser.net, 'is_jit_half', False), "denoiser.net must be JiT_half (use a /half model)"

    def remap(k):
        return ('net.model_dec.' + k[len('net.'):]) if k.startswith('net.') else k

    def src_of(new_name):
        return ('net.' + new_name[len('net.model_dec.'):]) if new_name.startswith('net.model_dec.') else new_name

    # Old JiT-H/16 patchified the full RGB image (B,3,H,W) with kernel=p/stride=p; the new model_dec
    # works on the pixel-unshuffled grid (B,3*p*p,H/p,W/p) with a 1x1 conv. The two are mathematically
    # equivalent, but x_embedder.proj1 has a different shape and final_layer.linear a different channel
    # order (old [ph,pw,c] vs pixel-unshuffle [c,ph,pw]), hence the explicit reshape / permute.
    p, zc, xc = int(denoiser.net.pixel_patch_size), int(denoiser.net.zc), int(denoiser.net.xc)
    rgb_compatible = (zc == 3 * p * p) and (xc == 3 * p * p)
    perm_final = None
    if rgb_compatible:
        idx = torch.arange(3 * p * p)
        c, ph, pw = idx // (p * p), (idx // p) % p, idx % p
        perm_final = (ph * (p * 3) + pw * 3 + c).long()

    def adapt(ok, ov, nk, dst_shape):
        if ok == 'net.x_embedder.proj1.weight' and rgb_compatible and tuple(ov.shape) == (dst_shape[0], 3, p, p):
            return ov.reshape(dst_shape[0], 3 * p * p, 1, 1), True
        if ok in ('net.final_layer.linear.weight', 'net.final_layer.linear.bias') and rgb_compatible and tuple(ov.shape) == tuple(dst_shape):
            return ov[perm_final].contiguous(), True
        return ov, False

    # --- 1) model state_dict ---
    cur = denoiser.state_dict()
    loaded, adapted_keys, skipped = [], [], []
    for ok, ov in ckpt['model'].items():
        nk = remap(ok)
        if nk not in cur:
            skipped.append((ok, nk, tuple(ov.shape), None))
            continue
        v, did = adapt(ok, ov, nk, tuple(cur[nk].shape))
        if v.shape == cur[nk].shape:
            cur[nk] = v
            loaded.append(nk)
            if did:
                adapted_keys.append(nk)
        else:
            skipped.append((ok, nk, tuple(ov.shape), tuple(cur[nk].shape)))
    denoiser.load_state_dict(cur)
    logger.info("load_jit_to_half[model]: loaded {}/{} (adapted {}), skipped {}", len(loaded), len(ckpt['model']), len(adapted_keys), len(skipped))
    for nk in adapted_keys:
        logger.info("  model adapt: {} (rgb<->grid remap)", nk)
    for ok, nk, ss, ts in skipped[:8]:
        logger.info("  model skip: {} -> {} (src {} vs dst {})", ok, nk, ss, ts)

    # --- 2) EMA ---
    def build_ema(old_sd):
        out = []
        for name, param in denoiser.named_parameters():
            src = src_of(name)
            if src in old_sd:
                ov = old_sd[src]
                v, _ = adapt(src, ov, name, tuple(param.shape))
                if v.shape == param.shape:
                    out.append(v.detach().clone().to(device))
                    continue
            out.append(param.detach().clone().to(device))
        return out
    denoiser.ema_params1 = build_ema(ckpt.get('model_ema1', {}))
    denoiser.ema_params2 = build_ema(ckpt.get('model_ema2', {}))
    logger.info("load_jit_to_half[ema]: rebuilt (fallback to current init where missing)")

    # --- 3) optimizer state (Adam moments migrated by parameter name) ---
    if optimizer is not None and 'optimizer' in ckpt and 'args' in ckpt:
        try:
            oa = copy.copy(ckpt['args'])
            old_den = Denoiser(oa)
            old_named = [(n, p) for n, p in old_den.named_parameters() if p.requires_grad]
            no_decay_names, decay_names = [], []
            for n, p in old_named:
                if len(p.shape) == 1 or n.endswith('.bias') or 'diffloss' in n:
                    no_decay_names.append(n)
                else:
                    decay_names.append(n)
            pg = ckpt['optimizer']['param_groups']
            assert len(pg) == 2, f"expected 2 param_groups, got {len(pg)}"
            assert len(pg[0]['params']) == len(no_decay_names), f"no_decay size mismatch: {len(pg[0]['params'])} vs {len(no_decay_names)}"
            assert len(pg[1]['params']) == len(decay_names), f"decay size mismatch: {len(pg[1]['params'])} vs {len(decay_names)}"
            id_to_name = {}
            for i, n in zip(pg[0]['params'], no_decay_names):
                id_to_name[i] = n
            for i, n in zip(pg[1]['params'], decay_names):
                id_to_name[i] = n
            old_state_by_name = {id_to_name[k]: v for k, v in ckpt['optimizer']['state'].items() if k in id_to_name}

            migrated, fresh = 0, 0
            for name, p in denoiser.named_parameters():
                if not p.requires_grad:
                    continue
                src = src_of(name)
                old_st = old_state_by_name.get(src)
                if old_st is None:
                    fresh += 1
                    continue
                ns = {}
                shape_ok = True
                for k, v in old_st.items():
                    if torch.is_tensor(v):
                        vv, _ = adapt(src, v, name, tuple(p.shape)) if v.shape != p.shape else (v, False)
                        if vv.shape == p.shape or vv.numel() <= 1:
                            ns[k] = vv.detach().clone().to(device)
                        else:
                            shape_ok = False
                            break
                    else:
                        ns[k] = v
                if shape_ok:
                    optimizer.state[p] = ns
                    migrated += 1
                else:
                    fresh += 1
            logger.info("load_jit_to_half[opt]: migrated {} params, {} fresh", migrated, fresh)
            del old_den
        except Exception as exc:
            logger.warning("load_jit_to_half[opt]: migration skipped ({})", exc)

    start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else None
    del ckpt
    return start_epoch


def main(args):
    misc.init_distributed_mode(args)
    configure_train_logger(args.output_dir)
    logger.info('Job directory: {}', os.path.dirname(os.path.realpath(__file__)))
    logger.opt(raw=True).info("Arguments:\n" + str(args).replace(', ', ',\n'))

    if args.tryrun:
        ob = args.batch_size
        args.batch_size = max(1, args.batch_size // max(1, args.tryrun_div))
        logger.info("Tryrun: batch_size {} -> {}", ob, args.batch_size)
        prev_steps = args.num_sampling_steps
        args.num_sampling_steps = 5
        logger.info("Tryrun: num_sampling_steps {} -> 5", prev_steps)

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms (no local lambda: spawn DataLoader workers must pickle dataset)
    transform_train = transforms.Compose([
        functools.partial(center_crop_arr, image_size=args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    if args.text_dim > 0 and args.blip3o_60k_path:
        from ldm_is_ae.data.blip3o60k import BLIP3o60kDataset
        dataset_train = BLIP3o60kDataset(
            tar_dir=args.blip3o_60k_path,
            image_size=args.img_size,
            shuffle=True,
            rank=global_rank,
            world_size=num_tasks,
        )
        sampler_train = None
        # workers=0: with workers>0 the ranks can desynchronise around epoch boundaries;
# workers=0 is slower but deterministic.
        dl_kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_mem, drop_last=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, **dl_kw)
        logger.info("Using BLIP3o60kDataset from {}", args.blip3o_60k_path)
    elif args.text_dim > 0 and args.blip3o_path:
        from ldm_is_ae.data.blip3o import BLIP3oLongDataset
        dataset_train = BLIP3oLongDataset(
            tar_dir=args.blip3o_path,
            image_size=args.img_size,
            shuffle=True,
            rank=global_rank,
            world_size=num_tasks,
        )
        sampler_train = None
        # IterableDataset + multi-worker duplicates data; keep single-process
        dl_kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_mem, drop_last=True)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, **dl_kw)
        logger.info("Using BLIP3oLongDataset from {}", args.blip3o_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        logger.info("{}", dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        logger.info("Sampler_train = {}", sampler_train)

        dl_kw = dict(sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        if args.num_workers > 0:
            dl_kw["multiprocessing_context"] = torch_mp.get_context("spawn")
        data_loader_train = torch.utils.data.DataLoader(dataset_train, **dl_kw)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    logger.info("Model = {}", model)
    if args.freeze_backbone:
        logger.info("freeze_backbone=ON: backbone frozen, only T2I text branch trainable")
        freeze_backbone(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: {:.6f}M", n_params / 1e6)

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size() * args.accum_iter
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("Base lr: {:.2e}", args.lr * 256 / eff_batch_size)
    logger.info("Actual lr: {:.2e}", args.lr)
    logger.info("Effective batch size: {}", eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module

    # half: encoder (net.model_enc) and decoder get separate groups carrying lr_scale; non-half uses plain grouping
    if getattr(getattr(model_without_ddp, "net", None), "is_jit_half", False):
        param_groups = misc.add_weight_decay_half_enc(model_without_ddp, args.weight_decay, enc_lr_scale=args.enc_lr_scale, dec_lr_scale=args.dec_lr_scale)
    else:
        param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    logger.info("{}", optimizer)

    global_step = 0

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_denoiser_checkpoint_model(model_without_ddp, checkpoint['model'])

        ema_state_dict1 = unwrap_torch_compile_net_keys(checkpoint['model_ema1'])
        ema_state_dict2 = unwrap_torch_compile_net_keys(checkpoint['model_ema2'])
        model_without_ddp.ema_params1 = [
            (ema_state_dict1[name].to(device=device, dtype=p.dtype).detach().clone()
             if name in ema_state_dict1 else p.detach().clone())
            for name, p in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            (ema_state_dict2[name].to(device=device, dtype=p.dtype).detach().clone()
             if name in ema_state_dict2 else p.detach().clone())
            for name, p in model_without_ddp.named_parameters()
        ]
        logger.info("Resumed checkpoint from {}", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if getattr(getattr(model_without_ddp, "net", None), "is_jit_half", False):
                for pg in optimizer.param_groups:
                    if "lr_scale" in pg:
                        pg["lr_scale"] = args.enc_lr_scale
                    if "dec_lr_scale" in pg:
                        pg["dec_lr_scale"] = args.dec_lr_scale
            global_step = checkpoint.get('global_step', 0) or 0
            logger.info("Loaded optimizer & scaler state! global_step={}", global_step)
        del checkpoint
    elif args.init_from_ckpt and os.path.exists(args.init_from_ckpt):
        load_init_from_ckpt(args.init_from_ckpt, model_without_ddp)
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        logger.info("Initialized model from {} (partial load, no optimizer state)", args.init_from_ckpt)
    elif args.init_jit_ckpt and os.path.exists(args.init_jit_ckpt):
        start_ep = load_jit_to_half_state(args.init_jit_ckpt, model_without_ddp, optimizer, device)
        if start_ep is not None:
            args.start_epoch = start_ep
        logger.info("Initialized /half model from non-half JiT checkpoint: {} (start_epoch={})", args.init_jit_ckpt, args.start_epoch)
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        logger.info("Training from scratch")

    # T2I: initialize text encoder and unconditional embeddings
    text_encoder = None
    if args.text_dim > 0:
        if args.text_encoder_path:
            from ldm_is_ae.data.text_encoder import Qwen3TextEncoder
            text_encoder = Qwen3TextEncoder(args.text_encoder_path)
            text_encoder.to(device)
            uncond_pooled = text_encoder.uncond_pooled().to(device)
            uncond_tokens = text_encoder.uncond_tokens().to(device)
            model_without_ddp.set_uncond(uncond_pooled, uncond_tokens)
            logger.info("Initialized text encoder from {} (text_dim={}, text_len={})", args.text_encoder_path, args.text_dim, args.text_len)
        else:
            logger.warning("text_dim>0 but no --text_encoder_path set; text encoder will be None")

    if not int(os.environ.get("TORCH_COMPILE_DISABLE", 0)):
        model_without_ddp.net = torch.compile(model_without_ddp.net)
        logger.info("denoiser.net wrapped with torch.compile")
    else:
        logger.info("torch.compile disabled (TORCH_COMPILE_DISABLE=1)")

    # Evaluate generation
    if args.evaluate_gen:
        logger.info("Evaluating checkpoint at {} epoch", args.start_epoch)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer, text_encoder=text_encoder)
        return

    if args.tryrun:
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(args.start_epoch)
        ep = args.start_epoch
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, ep, log_writer=log_writer, args=args, max_optimizer_steps=1, text_encoder=text_encoder)
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=ep, epoch_name="last")
        logger.info("Tryrun: saved checkpoint-last.pth after 1 optimizer step")
        ni0 = args.num_images
        if args.tryrun_num_images > 0:
            cap = min(ni0, args.tryrun_num_images)
            c = max(1, args.class_num)
            eff = min(max(cap, 1), c)
            args.num_images = max(1, (cap // eff) * eff)
        try:
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, ep, batch_size=args.gen_bsz, log_writer=log_writer, text_encoder=text_encoder)
            torch.cuda.empty_cache()
        finally:
            args.num_images = ni0
        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
        logger.info("Tryrun done, exit.")
        return

    # Training loop
    logger.info("Start training for {} epochs (max_total_optimizer_steps={})", args.epochs, args.max_total_optimizer_steps)
    start_time = time.time()
    trained_opt_steps = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        remaining = None
        if args.max_total_optimizer_steps > 0:
            remaining = max(1, args.max_total_optimizer_steps - trained_opt_steps)
        done = train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args,
                               max_optimizer_steps=remaining, text_encoder=text_encoder, global_step_start=global_step)
        global_step += (done or 0)
        trained_opt_steps += (done or 0)
        if args.max_total_optimizer_steps > 0 and trained_opt_steps >= args.max_total_optimizer_steps:
            logger.info("Reached max_total_optimizer_steps={} (cumulative {}); saving checkpoint-last and stopping",
                        args.max_total_optimizer_steps, trained_opt_steps)
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name="last")
            break

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 10 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and  epoch > 0 and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, text_encoder=text_encoder)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: {}", total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
