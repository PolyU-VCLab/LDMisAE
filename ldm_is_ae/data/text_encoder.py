"""
Text encoder wrapper for T2I training with JiT-half.

Follows DeCo's approach: Qwen3-1.7B.
Outputs full token sequences for in-context injection via self-attention.

Usage:
    encoder = Qwen3TextEncoder(weight_path="/path/to/Qwen3-1.7B")
    tokens = encoder(["a cat"])          # (1, 128, 2048) — full token seq
    pooled = encoder.encode_pooled(["a cat"])  # (1, 2048)
    uncond_tokens = encoder.uncond_tokens()    # (1, 128, 2048)
"""

import torch
import torch.nn as nn


class Qwen3TextEncoder(nn.Module):
    def __init__(self, weight_path, max_length=128):
        super().__init__()
        from transformers import Qwen3Model, Qwen2Tokenizer

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            weight_path, max_length=max_length, padding_side="right"
        )
        self.model = Qwen3Model.from_pretrained(weight_path).to(torch.float32)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_length = max_length

        # Cache unconditional encodings
        self.register_buffer("_uncond_pooled", None, persistent=False)
        self.register_buffer("_uncond_tokens", None, persistent=False)

    @torch.no_grad()
    def forward(self, prompts):
        """Encode prompts to full token sequences (B, L, D)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        if next(self.model.parameters()).is_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        y = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]  # (B, L, 2048)
        return torch.nan_to_num(y.float())

    @torch.no_grad()
    def encode_pooled(self, prompts):
        """Encode prompts to pooled embeddings (B, D). Takes last valid token."""
        y = self.forward(prompts)
        tokens = self.tokenizer(
            prompts, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        last_idx = tokens.attention_mask.sum(dim=1) - 1
        pooled = y[torch.arange(len(prompts), device=y.device), last_idx]
        return pooled.float()

    @torch.no_grad()
    def uncond_pooled(self):
        if self._uncond_pooled is None:
            self._uncond_pooled = self.encode_pooled([""])
        return self._uncond_pooled

    @torch.no_grad()
    def uncond_tokens(self):
        if self._uncond_tokens is None:
            self._uncond_tokens = self.forward([""])  # (1, L, D)
        return self._uncond_tokens


class IdentityTextEncoder(nn.Module):
    """Passthrough for already-embedded tensors."""
    def forward(self, prompts):
        return prompts

    def uncond_tokens(self):
        return torch.zeros(1, 1, 1)
