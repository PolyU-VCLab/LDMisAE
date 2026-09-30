import contextlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm_is_ae.losses.lpips import LPIPS as LPIPSLoss

from ldm_is_ae.backbone.dit import JiT_models
from ldm_is_ae.backbone.sit import SiT_half_models


def jit_checkpoint_model_sd(model_sd: dict, denoiser: nn.Module | None = None) -> dict:
    """Strip lpips_loss.* from a checkpoint when the model has no lpips submodule; keep it otherwise."""
    _has_lpips = denoiser is not None and getattr(denoiser, "lpips_loss", None) is not None
    if _has_lpips:
        return dict(model_sd)
    return {k: v for k, v in model_sd.items() if not str(k).startswith("lpips_loss.")}


def unwrap_torch_compile_net_keys(sd: dict) -> dict:
    """At save time `net` is torch.compile'd, so checkpoint keys are net._orig_mod.*; at resume/inference load time the module is not compiled yet, so keys are realigned to net.*."""
    pfx = "net._orig_mod."
    if not any(str(k).startswith(pfx) for k in sd):
        return sd
    return {("net." + str(k)[len(pfx):]) if str(k).startswith(pfx) else k: v for k, v in sd.items()}


def load_denoiser_checkpoint_model(denoiser: nn.Module, model_sd: dict):
    """Load with strict=False when the checkpoint has no lpips keys but the model enables LPIPS (the weights come from the LPIPS class pretrained init)."""
    sd = unwrap_torch_compile_net_keys(jit_checkpoint_model_sd(model_sd, denoiser))
    _lpips_missing = getattr(denoiser, "lpips_loss", None) is not None and not any(str(k).startswith("lpips_loss.") for k in sd)
    return denoiser.load_state_dict(sd, strict=not _lpips_missing)


def pixel_unshuffle_hw_to_grid(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, C * r^2, H//r, W//r); H and W must be divisible by r."""
    return F.pixel_unshuffle(x, downscale_factor)


def pixel_shuffle_grid_to_hw(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """(B, C * r^2, H, W) -> (B, C, H*r, W*r)。"""
    return F.pixel_shuffle(x, upscale_factor)


def rgb_full_to_grid(x: torch.Tensor, p: int) -> torch.Tensor:
    return pixel_unshuffle_hw_to_grid(x, p)


def grid_to_rgb_full(x: torch.Tensor, p: int, full_hw: int) -> torch.Tensor:
    """Grid (B, zc, H//p, W//p) -> RGB (B, 3, full_hw, full_hw)."""
    return pixel_shuffle_grid_to_hw(x, p)


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.zc = int(getattr(args, "zc", 3))
        self.pixel_patch_size = int(getattr(args, "pixel_patch_size", 16))
        p = self.pixel_patch_size
        self.full_img_size = int(args.img_size)
        assert self.full_img_size % p == 0, f"img_size {self.full_img_size} not divisible by pixel_patch_size {p}"
        self.grid = self.full_img_size // p
        # T2I: text conditioning
        self.text_dim = int(getattr(args, "text_dim", 0))
        self.text_len = int(getattr(args, "text_len", 128))

        half = "/half" in args.model
        net_kw = dict(
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            text_dim=self.text_dim,
            text_len=self.text_len,
        )
        if half:
            net_kw["enc_bp"] = int(getattr(args, "enc_bp", 1))
            net_kw["z_norm"] = getattr(args, "z_norm", None)
            # Eq.4 gamma(t)=t^k, the auxiliary-feature mixing exponent; None/missing -> 0.0 (behaviour before this flag existed)
            _pk = getattr(args, "placeholder_k", None)
            net_kw["placeholder_k"] = 0.0 if _pk is None else float(_pk)
            net_kw["input_size"] = self.full_img_size
            net_kw["patch_size"] = p
            if getattr(args, "depth_enc", None) is not None:
                net_kw["depth_enc"] = int(args.depth_enc)
            if getattr(args, "depth_dec", None) is not None:
                net_kw["depth_dec"] = int(args.depth_dec)
            if args.model in SiT_half_models:
                self.net = SiT_half_models[args.model](zc=self.zc, **net_kw)
            else:
                self.net = JiT_models[args.model](zc=self.zc, **net_kw)
        else:
            net_kw["input_size"] = self.grid
            net_kw["patch_size"] = 1
            net_kw["in_channels"] = self.zc
            self.net = JiT_models[args.model](**net_kw)
        assert hasattr(self.net, "forward_enc") and hasattr(self.net, "forward_dec"), (
            "Denoiser needs a */half model (e.g. JiT-B/half); plain JiT-B/16 has no forward_enc/forward_dec"
        )

        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.schedule_power = float(getattr(args, "schedule_power", 1.0))
        self.first_step_pr = bool(getattr(args, "first_step_pr", False))
        self.pr_steps = int(getattr(args, "pr_steps", 0))
        self.xp_ste = getattr(args, "xp_ste", False)

        self.v_mse_weight = float(getattr(args, "v_mse_weight", 2.0))
        self.lpips_weight = float(getattr(args, "lpips_weight", 0.1))
        self.xp_ste = getattr(args, "xp_ste", False)
        self.lpips_loss = None
        if self.lpips_weight != 0.0:
            _mp = getattr(args, "lpips_model_path", None)
            if not _mp:
                # LPIPS / VGG weights: when not passed explicitly, fall back to the taming layout in the repo root and in the cwd
                _rels = ("taming-transformers/taming/modules/autoencoder/lpips/vgg.pth",
                         "taming/modules/autoencoder/lpips/vgg.pth")
                _roots = (Path(__file__).resolve().parent.parent, Path.cwd())
                for _root in _roots:
                    for _rel in _rels:
                        _cand = _root / _rel
                        if _cand.is_file():
                            _mp = str(_cand)
                            break
                    if _mp:
                        break
                if not _mp:
                    raise ValueError(
                        "LPIPS weights not found; pass --lpips_model_path /path/to/vgg.pth "
                        "(or set lpips_weight=0 to turn the LPIPS term off). "
                        "Searched: " + ", ".join(str(r / rel) for r in _roots for rel in _rels))
            self.lpips_loss = LPIPSLoss(model_path=_mp).eval()
        # T2I: unconditional embeddings for CFG (token mode + pooled mode)
        self.register_buffer("_uncond_emb", None, persistent=False)
        self.register_buffer("_uncond_tokens", None, persistent=False)

    def set_uncond(self, pooled_emb, token_seq=None):
        """Store unconditional embeddings. token_seq: (1, L, D) for in-context text mode."""
        self._uncond_emb = pooled_emb.reshape(1, -1)
        if token_seq is not None:
            self._uncond_tokens = token_seq.reshape(1, -1, token_seq.shape[-1])

    def get_uncond_emb(self, device):
        return self._uncond_emb.to(device=device, dtype=torch.float32)

    def _is_text_token_mode(self, labels):
        return self.text_dim > 0 and labels is not None and labels.dim() == 3

    def drop_labels(self, labels):
        if self._is_text_token_mode(labels):
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            if drop.any():
                labels = labels.clone()
                uncond = self._uncond_tokens.to(device=labels.device, dtype=labels.dtype)
                uncond = uncond.expand(int(drop.sum()), -1, -1)
                labels[drop] = uncond
            return labels
        if self.text_dim > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            if drop.any():
                labels = labels.clone()
                uncond = self.get_uncond_emb(labels.device)
                labels[drop] = uncond
            return labels
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward_encoder(self, xs, labels, placeholder=None, t=None):
        assert xs.dim() == 4 and xs.shape[2] == xs.shape[3] == self.grid, (
            f"expected xs (N,C,{self.grid},{self.grid}), got {tuple(xs.shape)}"
        )
        if placeholder is None:
            ph = int(getattr(self.net, "placeholder_channels", self.zc))
            placeholder = torch.zeros(xs.size(0), ph, self.grid, self.grid, device=xs.device, dtype=xs.dtype)
        if t is None:
            t = torch.ones(xs.size(0), device=xs.device, dtype=torch.float32)
        else:
            t = t.reshape(-1).to(device=xs.device, dtype=torch.float32)
        return self.net.forward_enc(placeholder, xs, t, y=labels)

    def forward_decoder(self, z, labels, t=None):
        if t is None:
            t = torch.ones(z.size(0), device=z.device, dtype=torch.float32)
        else:
            t = t.reshape(-1)
        _, xp = self.net.forward_dec(z, t, y=labels)
        return xp

    def forward_ldm(self, zt, t, labels):
        """zt -> DiT-D(placeholder, xs) = xp -> DiT-E(placeholder, xp) yields zp; returns (zp, xp)."""
        t = t.reshape(-1).to(device=zt.device, dtype=torch.float32)
        ph, xp = self.net.forward_dec(zt, t, y=labels)

        # ── 8-bit STE quantization on xp (inference-time) ──
        if self.xp_ste:
            p = self.pixel_patch_size
            xp_rgb = pixel_shuffle_grid_to_hw(xp, p)
            xp_rgb = xp_rgb.clamp(-1, 1)
            xp_rgb_q = ((xp_rgb + 1) * 127.5).round() / 127.5 - 1
            xp_rgb = xp_rgb + (xp_rgb_q - xp_rgb).detach()
            xp = pixel_unshuffle_hw_to_grid(xp_rgb, p)

        zp = self.forward_encoder(xp, labels, placeholder=ph, t=t)
        return zp, xp

    def forward(self, x, labels):
        B = x.size(0)
        device = x.device
        p = self.pixel_patch_size
        # Shared encoding step: x -> x_u (pixel-unshuffle) -> z1 (detached afterwards, see the velocity target)
        t_enc = torch.ones(B, device=device, dtype=torch.float32)

        # === MAIN BRANCH ===
        self.net.model_enc.requires_grad_(False)
        t = self.sample_t(B, device=device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn(B, self.net.zc, x.shape[2], x.shape[3], device=device, dtype=x.dtype) * self.noise_scale
        tf = t.flatten()

        # ---> encode: x -> z1 (keeps grad, but z1.detach() is used as the velocity target)
        z1 = self.forward_encoder(x, labels, placeholder=None, t=t_enc)

        # ---> add noise + ldm: zt -> DiT-D -> xp(F) -> DiT-E -> zp
        labels_dropped = self.drop_labels(labels) if self.training else labels
        zt = t * z1.detach() + (1 - t) * e

        ph, xp = self.net.forward_dec(zt, tf, y=labels_dropped)

        # ── STE on xp ──
        if self.xp_ste:
            xp_rgb = pixel_shuffle_grid_to_hw(xp, p)
            xp_rgb = xp_rgb.clamp(-1, 1)
            xp_rgb_q = ((xp_rgb + 1) * 127.5).round() / 127.5 - 1
            xp_rgb = xp_rgb + (xp_rgb_q - xp_rgb).detach()
            xp_for_enc = pixel_unshuffle_hw_to_grid(xp_rgb, p)
        else:
            xp_for_enc = xp

        # ── encoder forward：t<0.5 frozen, t≥0.5 normal ──
        self.net.model_enc.requires_grad_(False)
        zp = self.forward_encoder(xp_for_enc, labels_dropped, placeholder=ph, t=tf)
        mask = tf >= 0.5
        if mask.any():
            self.net.model_enc.requires_grad_(True)
            zp_high = self.forward_encoder(xp_for_enc[mask], labels_dropped[mask],
                                           placeholder=ph[mask], t=tf[mask])
            zp[mask] = zp_high

        # ---> pixel loss
        denom = (1 - t).clamp_min(self.t_eps)
        w = 1.0 / (denom * denom)
        x_weight = w.flatten()
        x_loss = (xp - x) ** 2
        lp_per = x.new_zeros(B)
        if self.lpips_loss is not None:
            gt_rgb = grid_to_rgb_full(x, p, self.full_img_size)
            hat_rgb = grid_to_rgb_full(xp, p, self.full_img_size)
            _cm = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
            with _cm:
                _val = self.lpips_loss(hat_rgb.float(), gt_rgb.float())
                lp_per = torch.mean(_val, dim=(1, 2, 3)) * x_weight
        lp = lp_per.mean()

        # ---> ldm loss
        v = (z1.detach() - zt) / denom
        v_pred = (zp - zt.detach()) / denom
        v_loss = (v.detach() - v_pred) ** 2
        z_loss = (z1.detach() - zp) ** 2


        x_per_sample = (x_loss.mean(dim=(1, 2, 3)) * x_weight)
        x_mse = x_per_sample.mean()
        v_per = v_loss.mean(dim=(1, 2, 3))
        v_mse = v_per.mean()
        v_cohesion = v_per.var(unbiased=False)
        z_mse = z_loss.mean(dim=(1, 2, 3)).mean()
        lp_term = self.lpips_weight * lp
        loss = x_mse + lp_term + self.v_mse_weight * v_mse
        loss_per = x_per_sample + self.lpips_weight * lp_per + self.v_mse_weight * v_per

        return {
            "loss": loss,
            "x_mse": x_mse.detach(),
            "v_mse": v_mse.detach(),
            "v_cohesion": v_cohesion.detach(),
            "z_mse": z_mse.detach(),
            "lpips": lp.detach(),
            "lpips_term": lp_term.detach(),
            "x": x.detach(),
            "hatx": xp.detach(),
            "z1": z1.detach(),
            "hatz": zp.detach(),
            "v": v.detach(),
            "v_pred": v_pred.detach(),
            "t": t.detach(),
            "y": labels_dropped.detach(),
            "loss_per": loss_per.detach(),
        }

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, self.zc, self.grid, self.grid, device=device)
        lin = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        if self.schedule_power != 1.0:
            lin = 1.0 - (1.0 - lin) ** self.schedule_power
        timesteps = lin.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        pr = self.pr_steps if self.pr_steps > 0 else (1 if self.first_step_pr else 0)
        for i in range(pr):
            t, t_next = timesteps[i], timesteps[i + 1]
            zp, _ = self.forward_ldm(z, t, labels)
            new_noise = self.noise_scale * torch.randn_like(zp)
            z = t_next * zp + (1 - t_next) * new_noise

        for i in range(pr, self.steps - 1):
            t, t_next = timesteps[i], timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        x = self.forward_decoder(z, labels) if hasattr(self.net, "forward_dec") else z
        return grid_to_rgb_full(x, self.pixel_patch_size, self.full_img_size)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        x_cond, _ = self.forward_ldm(z, t, labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        if self._is_text_token_mode(labels):
            uncond = self._uncond_tokens.to(device=labels.device, dtype=labels.dtype)
            uncond = uncond.expand(labels.shape[0], -1, -1)
        elif self.text_dim > 0:
            uncond = self.get_uncond_emb(labels.device).expand_as(labels)
        else:
            uncond = torch.full_like(labels, self.num_classes)
        x_uncond, _ = self.forward_ldm(z, t, uncond)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
