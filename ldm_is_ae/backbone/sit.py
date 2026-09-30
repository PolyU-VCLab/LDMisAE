import math
import torch
import torch.nn as nn
from ldm_is_ae.backbone.sit_blocks import SiT, FinalLayer


class SiT_half_encdec(SiT):
    """SiT with configurable out_channels and corrected unpatchify for 2D grids."""
    def __init__(self, out_channels=None, **kwargs):
        hidden_size = int(kwargs.get("hidden_size", 1152))
        if out_channels is not None:
            kwargs.pop("out_channels", None)
        super().__init__(**kwargs)
        # Expand embedding table to num_classes+1 for Denoiser drop_labels
        old_emb = self.y_embedder.embedding_table
        if old_emb.num_embeddings == self.y_embedder.num_classes:
            new_emb = nn.Embedding(self.y_embedder.num_classes + 1, old_emb.embedding_dim)
            new_emb.weight.data[:self.y_embedder.num_classes] = old_emb.weight.data
            new_emb.weight.data[self.y_embedder.num_classes:] = old_emb.weight.data.mean(dim=0, keepdim=True)
            self.y_embedder.embedding_table = new_emb
        if out_channels is not None:
            oc = int(out_channels)
            self.out_channels = oc
            p = self.x_embedder.patch_size[0]
            self.final_layer = FinalLayer(hidden_size, p, oc)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        n = x.shape[1]
        h = w = int(math.isqrt(n))
        assert h * h == n, f"num_patches {n} not a perfect square for 2D feature map"
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)


class SiT_half(nn.Module):
    """SiT-half: decoder + encoder split (mirrors JiT_half interface)."""

    def __init__(self, zc=3, xc=None, depth_enc=2, depth_dec=26,
                 hidden_size=1152, num_heads=16, placeholder_k=0.0, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        p = int(kwargs.pop("patch_size"))
        full_in = int(kwargs.pop("input_size"))
        grid = full_in // p
        assert full_in % p == 0
        if xc is None:
            xc = 3 * p * p
        ph = int(xc)
        self.placeholder_channels = ph
        # Eq.4 gamma(t)=t^k, the mixing exponent (same semantics as JiT_half; 0 disables mixing)
        self.placeholder_k = float(placeholder_k)
        self.zc = zc
        self.xc = xc
        self.full_img_size = full_in
        self.pixel_patch_size = p
        self.grid = grid
        self.is_jit_half = True

        kw = dict(kwargs)
        kw.pop("in_channels", None)
        kw.pop("out_channels", None)

        self.model_dec = SiT_half_encdec(
            input_size=grid, patch_size=1,
            in_channels=int(zc), out_channels=int(ph) + int(xc),
            depth=depth_dec, hidden_size=hidden_size, num_heads=num_heads,
            learn_sigma=False, class_dropout_prob=0.0, **kw,
        )
        self.model_enc = SiT_half_encdec(
            input_size=grid, patch_size=1,
            in_channels=int(xc), out_channels=int(zc),
            depth=depth_enc, hidden_size=hidden_size, num_heads=num_heads,
            learn_sigma=False, class_dropout_prob=0.0, **kw,
        )

    def forward(self, z, t, y, return_xs=False):
        assert z.shape[1] == self.zc and z.shape[2] == z.shape[3] == self.grid, \
            f"expected z (N,{self.zc},{self.grid},{self.grid}), got {tuple(z.shape)}"
        placeholder, xs = self.forward_dec(z, t, y)
        zp = self.forward_enc(placeholder, xs, t, y)
        if return_xs:
            return zp, xs
        return zp

    def _placeholder_mix(self, placeholder, xs, t):
        """Eq.4: F_full = gamma(t)*F + (1-gamma(t))*F', gamma(t)=t^k (k=0 -> previous behaviour)."""
        n = xs.shape[0]
        tk = t.reshape(n, *([1] * (xs.dim() - 1))).to(device=xs.device, dtype=xs.dtype).pow(self.placeholder_k)
        return tk * xs + (1 - tk) * placeholder

    def forward_enc(self, placeholder, xs, t, y):
        n = xs.shape[0]
        return self.model_enc(self._placeholder_mix(placeholder, xs, t), t.reshape(n), y)

    def forward_dec(self, z, t, y):
        hat_x = self.model_dec(z, t.reshape(-1), y)
        placeholder, xs = hat_x[:, : self.placeholder_channels], hat_x[:, self.placeholder_channels :, :, :]
        return placeholder, xs


def SiT_XL_half(**kwargs):
    kwargs.setdefault("patch_size", 16)
    depth_enc = kwargs.pop("depth_enc", 2)
    depth_dec = kwargs.pop("depth_dec", 26)
    kwargs.pop("placeholder_channels", None)
    kwargs.pop("attn_drop", None)
    kwargs.pop("proj_drop", None)
    return SiT_half(zc=kwargs.pop("zc", 3), xc=kwargs.pop("xc", None),
                    placeholder_k=kwargs.pop("placeholder_k", 0.0),
                    depth_enc=depth_enc, depth_dec=depth_dec,
                    hidden_size=1152, num_heads=16, **kwargs)


SiT_half_models = {
    'SiT-XL/half': SiT_XL_half,
}
