# Copyright © 2026 Apple Inc.

"""Granite Switch: a Granite 4.x base with embedded, switch-routed aLoRA adapters.

Reference: https://github.com/generative-computing/granite-switch (HF backend,
``hf/modeling_granite_switch.py`` and ``hf/switch/multi.py``).

The upstream switch is a pair of parameter-free attention heads (a counting
head and a Kerdock-coded memory head) whose net effect is exact and simple:
every position is routed to the adapter of the most recent control token at
or before it, or to the base model (index 0) if none has been seen. Control
tokens are then rewritten to their substitute ids before embedding. This port
computes that routing directly, carrying it across calls in a ``SwitchCache``
(the two cache slots the upstream switch reserves become one routing trace).

Adapters are applied per token (aLoRA): tokens before the control token run on
the frozen base weights, so their KV entries are base-clean and reusable by
plain chat and by every adapter. ``lora_B`` is pre-scaled by alpha / rank in
the upstream checkpoint, so no runtime scaling is applied.

Not supported: Shadow Residual (``dual_stream``) checkpoints, MoE bases
(``num_local_experts > 0``) and the speech-to-text cascade (text only).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, SwitchCache
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    shared_intermediate_size: int
    logits_scaling: float
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    max_position_embeddings: int
    num_adapters: int
    adapter_ranks: List[int]
    adapter_token_ids: List[int]
    adapter_substitute_token_ids: Optional[List[int]] = None
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "qkv_proj",
            "o_proj",
            "shared_input_linear",
            "shared_output_linear",
        ]
    )
    projection_head_dim: Optional[int] = None
    attention_bias: bool = False
    hidden_act: str = "silu"
    num_local_experts: int = 0
    dual_stream: bool = False
    position_embedding_type: str = "rope"
    rope_theta: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = True
    # Upstream switch geometry; the routing it computes is reproduced exactly
    # without these, they are kept for config round-tripping only.
    switch_cache_layers: int = 2

    def __post_init__(self):
        if self.dual_stream:
            raise ValueError("Shadow Residual (dual_stream) checkpoints are not supported.")
        if self.num_local_experts > 0:
            raise ValueError("Granite Switch over an MoE base is not supported yet.")
        if self.position_embedding_type != "rope":
            raise ValueError("Only RoPE Granite Switch checkpoints are supported.")
        if self.rope_theta is None:
            params = self.rope_parameters or {}
            self.rope_theta = float(params.get("rope_theta", 10000.0))
            if self.rope_scaling is None and params.get("rope_type", "default") != "default":
                self.rope_scaling = params
        if len(self.adapter_token_ids) != self.num_adapters:
            raise ValueError(
                "Expected one control token per adapter (base-reset layouts are not supported)."
            )
        if self.adapter_substitute_token_ids is None:
            self.adapter_substitute_token_ids = list(self.adapter_token_ids)


class SwitchedLinear(nn.Module):
    """A (possibly fused) linear layer plus one LoRA per adapter.

    ``slices`` splits the output into independently adapted parts (q/k/v, or
    gate/up). Adapter ``k`` stores ``lora_a[k]`` as ``[len(slices) * r_k, in]``
    and ``lora_b[k][s]`` as ``[slices[s], r_k]``.
    """

    def __init__(self, in_dims: int, slices: List[int], ranks: List[int], bias: bool, lora: bool):
        super().__init__()
        self.slices = slices
        self.base = nn.Linear(in_dims, sum(slices), bias=bias)
        self.has_lora = lora
        if lora:
            n = len(slices)
            self.lora_a = [mx.zeros((n * r, in_dims)) for r in ranks]
            self.lora_b = [[mx.zeros((o, r)) for o in slices] for r in ranks]

    def __call__(self, x: mx.array, masks: Dict[int, mx.array]) -> mx.array:
        y = self.base(x)
        if not self.has_lora or not masks:
            return y
        for k, m in masks.items():
            a = self.lora_a[k - 1]
            bs = self.lora_b[k - 1]
            h = x @ a.T
            if len(bs) == 1:
                d = h @ bs[0].T
            else:
                hs = mx.split(h, len(bs), axis=-1)
                d = mx.concatenate([hi @ b.T for hi, b in zip(hs, bs)], axis=-1)
            y = y + d * m
        return y


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.projection_head_dim or dim // self.n_heads
        self.scale = args.attention_multiplier
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        targets = args.lora_target_modules
        self.qkv_proj = SwitchedLinear(
            dim, [q_size, kv_size, kv_size], args.adapter_ranks, args.attention_bias, "qkv_proj" in targets
        )
        self.o_proj = SwitchedLinear(
            q_size, [dim], args.adapter_ranks, args.attention_bias, "o_proj" in targets
        )
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )
        self._split = [q_size, q_size + kv_size]

    def __call__(self, x, masks, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = mx.split(self.qkv_proj(x, masks), self._split, axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out, masks)


class SharedMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.shared_intermediate_size
        targets = args.lora_target_modules
        self.input_linear = SwitchedLinear(
            dim, [hidden, hidden], args.adapter_ranks, False, "shared_input_linear" in targets
        )
        self.output_linear = SwitchedLinear(
            hidden, [dim], args.adapter_ranks, False, "shared_output_linear" in targets
        )

    def __call__(self, x, masks):
        gate, up = mx.split(self.input_linear(x, masks), 2, axis=-1)
        return self.output_linear(nn.silu(gate) * up, masks)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.shared_mlp = SharedMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.residual_multiplier = args.residual_multiplier

    def __call__(self, x, masks, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), masks, mask, cache) * self.residual_multiplier
        r = self.shared_mlp(self.post_attention_layernorm(h), masks)
        return h + r * self.residual_multiplier


def _trace_length(c) -> int:
    return c._idx if hasattr(c, "_idx") else c.offset


class GraniteSwitchModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        n_layers = args.num_hidden_layers - args.switch_cache_layers
        self.layers = [DecoderLayer(args) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        self._ctrl = mx.array(args.adapter_token_ids)
        self._subs = mx.array(args.adapter_substitute_token_ids)

    def route(self, inputs: mx.array, switch_cache=None):
        """Per-token adapter indices (0 = base) and control-token-exchanged ids."""
        B, L = inputs.shape
        match = inputs[..., None] == self._ctrl
        is_ctrl = match.any(axis=-1)
        which = mx.argmax(match, axis=-1)
        tokens = mx.where(is_ctrl, self._subs[which], inputs)
        expert = mx.where(is_ctrl, which + 1, 0)

        prev = mx.zeros((B,), dtype=mx.int32)
        if switch_cache is not None and switch_cache.keys is not None:
            n = _trace_length(switch_cache)
            if n > 0:
                prev = switch_cache.keys[:, 0, n - 1, 0].astype(mx.int32)

        pos = mx.where(is_ctrl, mx.arange(L)[None], -1)
        last = mx.cummax(pos, axis=1)
        idx = mx.where(
            last >= 0,
            mx.take_along_axis(expert, mx.maximum(last, 0), axis=1),
            prev[:, None],
        ).astype(mx.int32)
        return idx, tokens

    def _active_adapters(self, idx: mx.array, switch_cache, L: int):
        """Python-side set of adapters present in ``idx``.

        Prefill chunks (L > 1) are inspected directly. Single-token decode
        steps reuse the set recorded at the previous call, so the generation
        loop is not forced to synchronize on every token; the hint is
        invalidated whenever the cache length changes behind our back (trim,
        prompt-cache reuse, batch merge). Note: a control token *generated* by
        the model itself during decode is routed in the trace but its LoRA is
        only applied from the next prefill on.
        """
        hint = getattr(switch_cache, "_gs_hint", None) if switch_cache is not None else None
        n = _trace_length(switch_cache) if switch_cache is not None else None
        if L == 1 and hint is not None and hint[0] == n:
            active = hint[1]
        else:
            flags = mx.stack([(idx == k).any() for k in range(1, self.args.num_adapters + 1)])
            active = {k + 1 for k, f in enumerate(flags.tolist()) if f}
        return active

    def __call__(self, inputs: mx.array, cache=None):
        if cache is None:
            cache = [None] * (len(self.layers) + 1)
        layer_caches, switch_cache = cache[:-1], cache[-1]
        L = inputs.shape[1]

        idx, tokens = self.route(inputs, switch_cache)
        active = self._active_adapters(idx, switch_cache, L)

        h = self.embed_tokens(tokens) * self.embedding_multiplier
        masks = {k: (idx == k)[..., None].astype(h.dtype) for k in sorted(active)}

        if switch_cache is not None:
            trace = idx[:, None, :, None]
            switch_cache.update_and_fetch(trace, trace)
            switch_cache._gs_hint = (_trace_length(switch_cache), active)

        mask = create_attention_mask(h, layer_caches[0])
        for layer, c in zip(self.layers, layer_caches):
            h = layer(h, masks, mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteSwitchModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out / self.logits_scaling

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers] + [SwitchCache()]

    def sanitize(self, weights):
        """Map the upstream (HF) checkpoint layout to this module tree.

        Idempotent: already-converted MLX weights pass through unchanged.
        Upstream stores LoRA stacked over adapters and zero-padded to the
        largest rank (``[A, 1, max_rank, in]``); each adapter is trimmed to its
        own rank here, after checking the padding really is zero.
        """
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # The switch's codebook is the upstream routing engine's precomputed
        # code table; routing is computed exactly here without it.
        weights.pop("model.switch.codebook", None)
        lut = weights.pop("model.switch.control_to_substitute_lut", None)
        if lut is not None:
            for c, s in zip(self.args.adapter_token_ids, self.args.adapter_substitute_token_ids):
                if int(lut[c]) != s:
                    raise ValueError(f"control_to_substitute_lut[{c}] = {int(lut[c])}, config says {s}")

        ranks = self.args.adapter_ranks

        def per_adapter(t):
            # [A, 1, r_max, d] or [A, 1, d, r_max] -> list of [r_max, d] / [d, r_max]
            return [t[i, 0] for i in range(t.shape[0])]

        def trim_a(t, r):
            if t.shape[0] > r and mx.any(t[r:] != 0).item():
                raise ValueError("lora_A rank padding is not zero")
            return t[:r]

        def trim_b(t, r):
            if t.shape[1] > r and mx.any(t[:, r:] != 0).item():
                raise ValueError("lora_B rank padding is not zero")
            return t[:, :r]

        out = {}
        grouped = {}
        for nk, v in weights.items():
            if ".base_layer." in nk:
                out[nk.replace(".base_layer.", ".base.")] = v
                continue
            for tag in (".lora_A_slices.", ".lora_B_slices."):
                if tag in nk:
                    prefix, s = nk.split(tag)
                    grouped.setdefault(prefix, {}).setdefault(tag[1:7], {})[int(s)] = v
                    break
            else:
                if nk.endswith(".lora_A") or nk.endswith(".lora_B"):
                    prefix, kind = nk.rsplit(".", 1)
                    grouped.setdefault(prefix, {}).setdefault(kind, {})[0] = v
                else:
                    out[nk] = v

        for prefix, parts in grouped.items():
            a_slices = [parts["lora_A"][s] for s in sorted(parts["lora_A"])]
            b_slices = [parts["lora_B"][s] for s in sorted(parts["lora_B"])]
            a_per = [per_adapter(t) for t in a_slices]  # [slice][adapter]
            b_per = [per_adapter(t) for t in b_slices]
            for i, r in enumerate(ranks):
                out[f"{prefix}.lora_a.{i}"] = mx.concatenate(
                    [trim_a(a_per[s][i], r) for s in range(len(a_slices))], axis=0
                )
                for s in range(len(b_slices)):
                    out[f"{prefix}.lora_b.{i}.{s}"] = trim_b(b_per[s][i], r)
        return out
