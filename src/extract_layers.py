"""AntiDeepfake checkpoint loading and hidden-state extraction.

The published checkpoints are fairseq-style state dicts saved as safetensors
(``m_ssl.model.encoder.layers.0.self_attn.k_proj.weight`` ...) with a binary head
``proj_fc`` [2, hidden].  ``config.json`` however is a plain transformers
``Wav2Vec2ForPreTraining`` config, so the weights are remapped to the transformers
names instead of installing fairseq - which the TASKBOOK only allows as a 45-minute
budget item and which the X2 notes already flagged as a dependency trap.

Two details that bite (both already documented in ``wx2/docs/REPRODUCE.md``):

* ``layerdrop`` must be 0 at inference, otherwise the encoder silently returns
  21-23 hidden states instead of 49 and every layer index shifts;
* ``encoder.layers[i]``'s output is ``hidden_states[i+1]`` - hidden state 0 is the
  feature-projection output, not the first transformer block.

Only feature extraction is implemented here; ``proj_fc`` gives the zero-shot
real/fake logits and its class order is determined empirically (and recorded),
never assumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16000
N_SAMPLES = 64600          # 4.0375 s, the ML-ITW / SSL_Anti-spoofing protocol
MODEL_PREFIX = "m_ssl.model."

__all__ = [
    "SAMPLE_RATE", "N_SAMPLES", "MODEL_PREFIX",
    "remap_state_dict", "load_antideepfake", "prepare_waveform",
    "AntiDeepfakeExtractor", "STD_LAYER_FRACTIONS",
]

# ~8 standard-deviation layers: 0.25L, 0.5L, 0.625L, 0.75L, 0.875L, L-1, L
# (L = number of transformer layers), plus the fractions Line A used.
STD_LAYER_FRACTIONS = (0.25, 0.5, 0.625, 0.75, 0.875)


def _remap_one(key: str) -> Optional[str]:
    """fairseq name -> transformers name, or None for a pre-training-only tensor."""
    if not key.startswith(MODEL_PREFIX):
        return None                     # proj_fc lives outside the SSL model
    k = key[len(MODEL_PREFIX):]
    if k.startswith("feature_extractor.conv_layers."):
        parts = k.split(".")
        layer, rest = parts[2], parts[3:]
        if rest[0] == "0":              # conv
            return f"wav2vec2.feature_extractor.conv_layers.{layer}.conv.{rest[-1]}"
        if rest[0] == "2":              # layer norm (present from layer 1 on)
            return f"wav2vec2.feature_extractor.conv_layers.{layer}.layer_norm.{rest[-1]}"
        return None
    if k.startswith("post_extract_proj."):
        return "wav2vec2.feature_projection.projection." + k.split(".")[-1]
    if k.startswith("layer_norm."):
        return "wav2vec2.feature_projection.layer_norm." + k.split(".")[-1]
    if k.startswith("encoder.pos_conv.0."):
        return "wav2vec2.encoder.pos_conv_embed.conv." + k.split(".")[-1]
    if k.startswith("encoder.layer_norm."):
        return "wav2vec2.encoder.layer_norm." + k.split(".")[-1]
    if k.startswith("encoder.layers."):
        parts = k.split(".")
        layer, rest = parts[2], ".".join(parts[3:])
        rest = rest.replace("self_attn_layer_norm.", "attention_layer_norm.")
        rest = rest.replace("self_attn.", "attention.")
        rest = rest.replace("fc1.", "feed_forward.intermediate_dense.")
        rest = rest.replace("fc2.", "feed_forward.output_dense.")
        return f"wav2vec2.encoder.layers.{layer}.{rest}"
    # quantizer / project_q / mask_emb / final_proj: pre-training heads, unused
    return None


def remap_state_dict(sd: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a fairseq-style checkpoint into (encoder weights, extra heads)."""
    enc: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in sd.items():
        new = _remap_one(k)
        if new is None:
            extra[k] = v
        else:
            enc[new] = v
    return enc, extra


def load_antideepfake(model_dir, device: str = "cuda", dtype="float32"):
    """Load the remapped encoder plus the ``proj_fc`` binary head."""
    import torch
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    # .to("float32") is read as a DEVICE string; resolve the dtype properly
    if isinstance(dtype, str):
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[dtype]
    model_dir = Path(model_dir)
    cfg = Wav2Vec2Config.from_pretrained(str(model_dir))
    cfg.layerdrop = 0.0                 # otherwise the encoder randomly drops layers
    raw = load_file(str(model_dir / "model.safetensors"))
    enc, extra = remap_state_dict(raw)
    model = Wav2Vec2Model(cfg)
    missing, unexpected = model.load_state_dict(enc, strict=False)
    head_w = extra.get("proj_fc.weight")
    head_b = extra.get("proj_fc.bias")
    info = {
        "n_encoder_tensors": len(enc),
        "n_missing": len(missing),
        "missing_sample": list(missing)[:10],
        "n_unexpected": len(unexpected),
        "unexpected_sample": list(unexpected)[:10],
        "has_head": head_w is not None and head_b is not None,
        "head_shape": list(head_w.shape) if head_w is not None else None,
        "n_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "do_stable_layer_norm": bool(cfg.do_stable_layer_norm),
        "layerdrop": float(cfg.layerdrop),
    }
    model = model.to(device).to(dtype).eval()
    head = None
    if head_w is not None:
        head = (head_w.to(device).to(dtype), head_b.to(device).to(dtype))
    return model, cfg, head, info


def prepare_waveform(x: np.ndarray, target: int = N_SAMPLES) -> np.ndarray:
    """Mono, first ``target`` samples, ``np.tile`` padding for short clips.

    Zero padding is explicitly not used (Line A / SSL_Anti-spoofing protocol).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(target, np.float32)
    if x.size >= target:
        return x[:target]
    reps = int(np.ceil(target / x.size))
    return np.tile(x, reps)[:target]


class AntiDeepfakeExtractor:
    """Batched hidden-state + zero-shot extraction with a padding mask."""

    def __init__(self, model_dir, device: str = "cuda", dtype="float32",
                 waveform_layer_norm: bool = True) -> None:
        import torch

        self.torch = torch
        self.model, self.cfg, self.head, self.info = load_antideepfake(model_dir, device, dtype)
        self.device = device
        self.dtype = dtype
        self.waveform_layer_norm = waveform_layer_norm
        self.n_layers = int(self.cfg.num_hidden_layers)
        self.std_layers = sorted({
            min(self.n_layers, max(1, int(round(f * self.n_layers)))) for f in STD_LAYER_FRACTIONS
        } | {self.n_layers - 1, self.n_layers})

    def _batch(self, waves: Sequence[np.ndarray]):
        torch = self.torch
        x = torch.tensor(np.stack(waves), device=self.device, dtype=torch.float32)
        if self.waveform_layer_norm:
            x = torch.nn.functional.layer_norm(x, x.shape[1:])
        return x.to(self.dtype)

    def extract(self, waves: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        """Return frame-mean of every hidden state, std of ``std_layers``, zero-shot logits."""
        torch = self.torch
        x = self._batch(waves)
        with torch.no_grad():
            out = self.model(x, output_hidden_states=True)
        hs = out.hidden_states                 # tuple of (B, T', D); hs[0] = projection
        mean = torch.stack([h.mean(dim=1) for h in hs], dim=1)     # (B, L+1, D)
        std = torch.stack([hs[i].std(dim=1) for i in self.std_layers], dim=1)
        zs = None
        if self.head is not None:
            pooled = hs[-1].mean(dim=1)
            w, b = self.head
            zs = pooled.to(w.dtype) @ w.t() + b
        res = {
            "mean": mean.float().cpu().numpy().astype(np.float16),
            "std": std.float().cpu().numpy().astype(np.float16),
            "n_hidden_states": np.array([len(hs)]),
        }
        if zs is not None:
            res["head_logits"] = zs.float().cpu().numpy()
        return res
