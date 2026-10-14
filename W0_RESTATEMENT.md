# W0 restatement (TASKBOOK 1.2)

- utc: 2026-10-14T18:16:07Z   host: autodl-container-8r96hvx588-6d726b57
- platform x label counts match the frozen X2 table: **False** (total (18168, 6361, 24529))

## 1. cell filter rule

A cell is platform|language (29 of them). A cell is only eligible to donate labels when it holds more than k+20 clips; smaller cells contribute no labels and fall back to their platform rate. Label counts per cell are therefore k=10 per cell at k=290, not 290/29 rounded.

read from: `wx2/scripts/fsc_var.py (len(idx) > k + 20), wx2/scripts/p0_fewshot.py`

## 2. is the label draw speaker-restricted?

No. Clips are drawn with rng.choice(idx, k, replace=False) per cell with no speaker constraint, so the same speaker can appear in L and in E. X3 adds the strict mask precisely to measure how much that matters.

read from: `wx2/scripts/fsc_var.py, wx2/scripts/p0_fewshot.py`

## 3. exact definition of the clips / strict masks

clips = every unlabelled clip in the region. strict = additionally drop evaluation clips whose SPEAKER also appears in L. Line A reports both as 13.64 pooled at k=290, and W4 checks whether they are bit-identical; X3 implements the strict mask as an explicit speaker-disjoint filter (src/draws.py::strict_eval_mask) instead of relying on that coincidence.

read from: `wx2/docs/COMPARISON.md E.5; X3 src/draws.py`

## 4. probe structure and hyper-parameters

An MLP probe over layers [8,12,16,18,20] of the 25 mean-pooled layer vectors: LayerNorm of the flattened input, Linear(flat_dim, 512), GELU, then the output unit; trained with the script's fixed recipe (not a CV grid). The per-clip variant additionally z-scores each layer per clip (clip_z_layer).

read from: `wx2/scripts/train_corpus.py (LAYERS, class Net, clip_z_layer)`

## 5. which layer and pooling the features use

All 25 hidden states of facebook/wav2vec2-xls-r-300m are kept, each mean-pooled over time, after decoding to mono 16 kHz and taking the FIRST 64,600 samples (4.0375 s) with np.tile padding for short clips (never zero padding). Stored as (24529, 25, 1024) float16 in mlitw_feats.npy.

read from: `wx2/scripts/extract_ssl.py (SR, N, Wav2Vec2Model hidden states, mean over time)`

## Inventory

- **hostname**: `autodl-container-8r96hvx588-6d726b57`
- **utc**: `2026-10-14T18:16:07Z`
- **nproc**: `208`
- **memory**: `total        used        free      shared  buff/cache   available
Mem:            1007          95         347          14         564         891
Swap:              0           0           0`
- **disk**: `Filesystem      Size  Used Avail Use% Mounted on
/dev/md0        220G  201G   20G  92% /root/autodl-tmp
overlay          30G  6.9G   24G  23% /`
- **gpu**: `name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
NVIDIA GeForce RTX 4080 SUPER, 0 %, 0 MiB, 32760 MiB`
- **ffmpeg**: `/usr/bin/ffmpeg`
- **tmux**: `/usr/bin/tmux`
- **python**: `3.12.3`
