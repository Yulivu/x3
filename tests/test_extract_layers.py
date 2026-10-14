"""Pins for the checkpoint remapping and the waveform policy (no weights needed)."""

import numpy as np

from src.extract_layers import (
    MODEL_PREFIX,
    N_SAMPLES,
    prepare_waveform,
    remap_state_dict,
)


def _fake_fairseq_sd(n_layers=2, hidden=8, conv_dim=(4, 4)):
    sd = {}
    for i, c in enumerate(conv_dim):
        sd[f"{MODEL_PREFIX}feature_extractor.conv_layers.{i}.0.weight"] = np.zeros((c, 1, 3))
        sd[f"{MODEL_PREFIX}feature_extractor.conv_layers.{i}.0.bias"] = np.zeros(c)
        if i > 0:
            sd[f"{MODEL_PREFIX}feature_extractor.conv_layers.{i}.2.1.weight"] = np.zeros(c)
            sd[f"{MODEL_PREFIX}feature_extractor.conv_layers.{i}.2.1.bias"] = np.zeros(c)
    sd[f"{MODEL_PREFIX}post_extract_proj.weight"] = np.zeros((hidden, conv_dim[-1]))
    sd[f"{MODEL_PREFIX}post_extract_proj.bias"] = np.zeros(hidden)
    sd[f"{MODEL_PREFIX}layer_norm.weight"] = np.zeros(hidden)
    sd[f"{MODEL_PREFIX}layer_norm.bias"] = np.zeros(hidden)
    sd[f"{MODEL_PREFIX}encoder.pos_conv.0.weight_g"] = np.zeros((hidden, 1, 1))
    sd[f"{MODEL_PREFIX}encoder.pos_conv.0.weight_v"] = np.zeros((hidden, hidden, 1))
    sd[f"{MODEL_PREFIX}encoder.pos_conv.0.bias"] = np.zeros(hidden)
    sd[f"{MODEL_PREFIX}encoder.layer_norm.weight"] = np.zeros(hidden)
    sd[f"{MODEL_PREFIX}encoder.layer_norm.bias"] = np.zeros(hidden)
    for i in range(n_layers):
        pre = f"{MODEL_PREFIX}encoder.layers.{i}."
        for name in ("k_proj", "v_proj", "q_proj", "out_proj"):
            sd[pre + f"self_attn.{name}.weight"] = np.zeros((hidden, hidden))
            sd[pre + f"self_attn.{name}.bias"] = np.zeros(hidden)
        sd[pre + "self_attn_layer_norm.weight"] = np.zeros(hidden)
        sd[pre + "self_attn_layer_norm.bias"] = np.zeros(hidden)
        sd[pre + "fc1.weight"] = np.zeros((hidden * 2, hidden))
        sd[pre + "fc1.bias"] = np.zeros(hidden * 2)
        sd[pre + "fc2.weight"] = np.zeros((hidden, hidden * 2))
        sd[pre + "fc2.bias"] = np.zeros(hidden)
        sd[pre + "final_layer_norm.weight"] = np.zeros(hidden)
        sd[pre + "final_layer_norm.bias"] = np.zeros(hidden)
    # pre-training-only tensors that must be dropped
    sd[f"{MODEL_PREFIX}quantizer.weight_proj.weight"] = np.zeros((4, hidden))
    sd[f"{MODEL_PREFIX}project_q.weight"] = np.zeros((hidden, hidden))
    sd[f"{MODEL_PREFIX}mask_emb"] = np.zeros((1, 1, hidden))
    sd[f"{MODEL_PREFIX}final_proj.weight"] = np.zeros((hidden, hidden))
    # the downstream head lives outside the SSL prefix
    sd["proj_fc.weight"] = np.zeros((2, hidden))
    sd["proj_fc.bias"] = np.zeros(2)
    return sd


def test_remap_covers_every_needed_tensor_and_drops_pretraining_heads():
    enc, extra = remap_state_dict(_fake_fairseq_sd())
    # Wav2Vec2Model wants these WITHOUT a "wav2vec2." prefix (that prefix belongs to
    # the ForPreTraining / ForCTC wrappers)
    assert "feature_extractor.conv_layers.0.conv.weight" in enc
    assert "feature_extractor.conv_layers.1.layer_norm.weight" in enc
    assert "feature_projection.projection.weight" in enc
    assert "feature_projection.layer_norm.weight" in enc
    assert "encoder.pos_conv_embed.conv.weight_g" in enc
    assert "encoder.layers.0.attention.k_proj.weight" in enc
    assert "encoder.layers.0.attention_layer_norm.weight" in enc
    assert "encoder.layers.0.feed_forward.intermediate_dense.weight" in enc
    assert "encoder.layers.0.feed_forward.output_dense.weight" in enc
    assert "encoder.layers.1.final_layer_norm.bias" in enc
    assert not any(k.startswith("wav2vec2.") for k in enc)
    # both spellings of the two version-dependent names are emitted
    assert "encoder.layers.0.attention_layer_norm.weight" in enc
    assert "encoder.layers.0.layer_norm.weight" in enc
    assert "encoder.pos_conv_embed.conv.weight_g" in enc
    assert "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in enc
    assert "encoder.pos_conv_embed.conv.parametrizations.weight.original1" in enc
    # nothing fairseq-shaped survives, and no pre-training head leaks through
    assert not any(k.startswith(MODEL_PREFIX) for k in enc)
    assert not any("quantizer" in k or "project_q" in k or "mask_emb" in k
                   or "final_proj" in k for k in enc)
    assert "proj_fc.weight" in extra and "proj_fc.bias" in extra


def test_prepare_waveform_tiles_short_clips_and_truncates_long_ones():
    short = np.arange(1000, dtype=np.float32)
    out = prepare_waveform(short)
    assert out.shape == (N_SAMPLES,)
    # tiling, not zero padding: the result is exactly np.tile(x, ...)[:N]
    expected = np.tile(short, int(np.ceil(N_SAMPLES / short.size)))[:N_SAMPLES]
    assert np.array_equal(out, expected)
    long = np.arange(N_SAMPLES * 2, dtype=np.float32)
    assert np.array_equal(prepare_waveform(long), long[:N_SAMPLES])
    assert prepare_waveform(np.zeros(0, np.float32)).shape == (N_SAMPLES,)
