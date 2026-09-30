# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ldm_is_ae.utils.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        zero_final_linear=True,
        text_dim=0,
        text_len=128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.zero_final_linear = bool(zero_final_linear)
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.text_len = text_len

        # time and text/class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        if text_dim > 0:
            self.y_embedder = nn.Sequential(
                nn.Linear(text_dim, hidden_size),
                RMSNorm(hidden_size, eps=1e-6),
            )
            # text token position embedding
            self.text_pos_embed = nn.Parameter(
                torch.zeros(1, text_len, hidden_size), requires_grad=True
            )
            torch.nn.init.normal_(self.text_pos_embed, std=.02)
            # Override in_context_{len,start} for text token injection
            self.in_context_len = text_len
            self.in_context_start = 0
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token (class mode only; text mode injects real text tokens)
        if self.in_context_len > 0 and text_dim == 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize text token position embedding:
        if self.text_dim > 0:
            nn.init.normal_(self.text_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label/text embedding:
        if self.text_dim > 0:
            nn.init.xavier_uniform_(self.y_embedder[0].weight)
            nn.init.constant_(self.y_embedder[0].bias, 0)
        else:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        if self.zero_final_linear:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # time embedding
        t_emb = self.t_embedder(t)

        # Text token mode: y is (B, L, D) full token sequence
        text_token_mode = self.text_dim > 0 and y.dim() == 3
        if text_token_mode:
            text_tokens = self.y_embedder(y)  # (B, L, H)
            text_tokens = text_tokens + self.text_pos_embed[:, : y.shape[1]]
            c = t_emb  # adaLN from timestep only; text is injected via in-context
            y_emb = None
        else:
            y_emb = self.y_embedder(y)
            c = t_emb + y_emb
            text_tokens = None

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                if text_token_mode:
                    x = torch.cat([text_tokens, x], dim=1)
                else:
                    in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                    in_context_tokens += self.in_context_posemb
                    x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output


class RMSNormNoScale(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.dims = tuple(range(-len(self.normalized_shape), 0))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        x = xf * torch.rsqrt(xf.pow(2).mean(dim=self.dims, keepdim=True) + self.eps)
        return x.to(dtype)


class JiT_half(nn.Module):
    """Two-stage JiT: the decoder takes only the latent z (in=zc) and outputs placeholder+xc; the encoder
        takes the placeholder / xs mixed features."""

    def __init__(self, zc=3, xc=None, depth_enc=12, depth_dec=12, placeholder_k=0.0, **kwargs):
        super().__init__()
        kwargs = dict(kwargs)
        kwargs.pop("depth", None)
        p = int(kwargs["patch_size"])
        full_in = int(kwargs["input_size"])
        grid = full_in // p
        assert full_in % p == 0, f"input_size {full_in} not divisible by patch_size {p}"
        if xc is None:
            xc = 3 * p * p
        ph = int(xc)
        self.placeholder_channels = ph
        # Eq.4 gamma(t)=t^k, the time-aware auxiliary-feature mixing exponent (the paper uses k=3; 0 disables mixing)
        self.placeholder_k = float(placeholder_k)
        self.enc_bp = bool(kwargs.pop("enc_bp", True))
        self.z_norm_spec = kwargs.pop("z_norm", None)  # None / "rms": normalize the model_enc output (option B, no learnable params)
        kw = dict(kwargs)
        kw.pop("in_channels", None)
        kw.pop("out_channels", None)
        kw["input_size"] = grid
        kw["patch_size"] = 1
        _icl, _ics = int(kw.get("in_context_len", 0)), int(kw.get("in_context_start", 0))
        if _icl > 0 and depth_dec <= _ics:
            raise ValueError(f"JiT_half: depth_dec ({depth_dec}) must be > in_context_start ({_ics}); use at least {_ics + 1}")
        self.zc = zc
        self.xc = xc
        self.full_img_size = full_in
        self.pixel_patch_size = p
        self.grid = grid
        self.z_norm = None
        if self.z_norm_spec in ("rms", "RMSNormNoScale"):
            # Option B: reuse RMSNormNoScale (no learnable parameters) to normalise the model_enc output
            # per sample over (zc,H,W)
            self.z_norm = RMSNormNoScale(normalized_shape=(self.zc, self.grid, self.grid))
        self.is_jit_half = True
        kw_dec = dict(kw)
        kw_dec["in_channels"] = int(zc)
        kw_dec["out_channels"] = int(ph) + int(xc)
        kw_dec["text_dim"] = int(kw.get("text_dim", 0))
        kw_dec["text_len"] = int(kw.get("text_len", 128))
        kw_enc = dict(kw)
        if _icl > 0:
            kw_enc["in_context_start"] = 0
        kw_enc["in_channels"] = int(xc)
        kw_enc["out_channels"] = int(zc)
        kw_enc["zero_final_linear"] = False
        self.model_dec = JiT(depth=depth_dec, **kw_dec)
        self.model_enc = JiT(depth=depth_enc, **kw_enc)

    def forward(self, z, t, y, return_xs=False):
        assert z.shape[1] == self.zc and z.shape[2] == z.shape[3] == self.grid, f"expected z (N,{self.zc},{self.grid},{self.grid}), got {tuple(z.shape)}"
        placeholder, xs = self.forward_dec(z, t, y)
        zp = self.forward_enc(placeholder, xs, t, y)
        if return_xs:
            return zp, xs
        return zp

    def _placeholder_mix(self, placeholder, xs, t):
        """Eq.4 time-aware auxiliary-feature mixing: F_full = gamma(t)*F + (1-gamma(t))*F', gamma(t)=t^k.
            k=0 -> gamma is always 1 -> F_full = xs, i.e. the encoder input carries no auxiliary
            feature F' (behaviour before this flag existed)."""
        n = xs.shape[0]
        tk = t.reshape(n, *([1] * (xs.dim() - 1))).to(device=xs.device, dtype=xs.dtype).pow(self.placeholder_k)
        return tk * xs + (1 - tk) * placeholder

    def forward_enc(self, placeholder, xs, t, y):
        n = placeholder.shape[0]
        B, _, H, W = xs.shape
        if self.enc_bp:
            bp = F.interpolate(xs.flatten(2).transpose(1, 2), size=self.zc, mode="linear", align_corners=False)
            bp = bp.transpose(1, 2).view(B, self.zc, H, W)
        else:
            bp = None
        zx = self._placeholder_mix(placeholder, xs, t)   # Eq.4: gamma(t)*xs + (1-gamma(t))*placeholder
        raw = self.model_enc(zx, t.reshape(n), y)
        if self.z_norm is not None:
            raw = self.z_norm(raw)
        return raw if bp is None else raw + bp

    def forward_dec(self, z, t, y):
        hat_x = self.model_dec(z, t, y)
        placeholder, xs = hat_x[:, : self.placeholder_channels], hat_x[:, self.placeholder_channels :, :, :]
        return placeholder, xs


def JiT_B_half(**kwargs):
    kwargs.setdefault("patch_size", 16)
    depth_enc = kwargs.pop("depth_enc", 6)
    depth_dec = kwargs.pop("depth_dec", 6)
    kwargs.pop("placeholder_channels", None)
    return JiT_half(zc=kwargs.pop("zc", 3), xc=kwargs.pop("xc", None), placeholder_k=kwargs.pop("placeholder_k", 0.0), depth_enc=depth_enc, depth_dec=depth_dec, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, **kwargs)


def JiT_L_half(**kwargs):
    kwargs.setdefault("patch_size", 16)
    depth_enc = kwargs.pop("depth_enc", 12)
    depth_dec = kwargs.pop("depth_dec", 12)
    kwargs.pop("placeholder_channels", None)
    return JiT_half(zc=kwargs.pop("zc", 3), xc=kwargs.pop("xc", None), placeholder_k=kwargs.pop("placeholder_k", 0.0), depth_enc=depth_enc, depth_dec=depth_dec, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, **kwargs)


def JiT_H_half(**kwargs):
    kwargs.setdefault("patch_size", 16)
    depth_enc = kwargs.pop("depth_enc", 16)
    depth_dec = kwargs.pop("depth_dec", 16)
    kwargs.pop("placeholder_channels", None)
    return JiT_half(zc=kwargs.pop("zc", 3), xc=kwargs.pop("xc", None), placeholder_k=kwargs.pop("placeholder_k", 0.0), depth_enc=depth_enc, depth_dec=depth_dec, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, **kwargs)


def JiT_B_16(**kwargs):
    kwargs.setdefault("patch_size", 16)
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, **kwargs)

def JiT_B_32(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return JiT(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, in_context_len=32, in_context_start=4, **kwargs)

def JiT_L_16(**kwargs):
    kwargs.setdefault("patch_size", 16)
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, **kwargs)

def JiT_L_32(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return JiT(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, in_context_len=32, in_context_start=8, **kwargs)

def JiT_H_16(**kwargs):
    kwargs.setdefault("patch_size", 16)
    return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, **kwargs)

def JiT_H_32(**kwargs):
    kwargs.setdefault("patch_size", 32)
    return JiT(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, in_context_len=32, in_context_start=10, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
    'JiT-B/half': JiT_B_half,
    'JiT-L/half': JiT_L_half,
    'JiT-H/half': JiT_H_half,
}
