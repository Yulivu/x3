# X3 report

```
DECISION: G0=pass G1=HEADROOM_OK G2=REPRODUCED G3=xls-r-2b|S3 G4=C1_FAIL G5=PL_OK G6=PAST_FAIL G7=CASA_PASS G8=Z_FAIL G9=NOT_RUN G10=DECONF_NOT_SHOWN G11=SOTA_F_DEV_NO G12=MALP_INFO G13=SEL_FAIL G14=SPEAKER_COVERAGE_FAIL G15=LABEL_EFFICIENCY_PASS G16=SOTA_F2_DEV_NO G17=ESN_FAIL G18=PROJ_FAIL
```
FLAGS: SPEAKER_GAP=False LP_GAIN_IS_SPEAKER_LOCAL=False B11_BROKEN=True IDENTITY_OVERLAP_FLAG=False

## 1. Summary

- S-F (few-shot SOTA): SOTA_F_DEV_NO
- S-Z (zero-label SOTA): Z_FAIL
- N1 (MPP / C1): C1_FAIL
- N2 (PAST): PAST_FAIL
- N3 (CASA): CASA_PASS
- N4 (de-confounding): DECONF_NOT_SHOWN
- S-K (budget curve): see results/w7_curve.csv
- S-R (ITW replication): NOT_RUN

Novelty rule: 1/4 of N1-N4 passed -> one component passed: report it as an implementation detail.

## NEW_TASKS 02 (v3 protocol, priority over U4)

### U1' main table - pooled EER on the v3 dev region (100 draws)

`results/u1p_speaker_split_v3.csv` (rows: budget x mask)

| k | mask | B1 | B3 | B7 | B9 | C1 |
|---|---|---|---|---|---|---|
| 100 | seen | 11.75 | 12.68 | 11.46 | 10.18 | 11.72 |
| 100 | strict_draw | 15.73 | 18.08 | 16 | 18.84 | 17.16 |
| 100 | unseen | 15.13 | 18.81 | 15.59 | 16.39 | 16.83 |
| 290 | seen | 10.19 | 11.8 | 9.705 | 7.249 | 10.13 |
| 290 | strict_draw | 14.45 | 17.15 | 15.13 | 15.04 | 16.1 |
| 290 | unseen | 11.96 | 17.36 | 13.88 | 10.77 | 14.84 |

Intervals at k=290: old rule (2.5/97.5 of the per-draw differences) vs the speaker-clustered (b) interval

| mask | method_a | method_b | delta | old_lo | old_hi | b_lo | b_hi | b_contains_zero |
|---|---|---|---|---|---|---|---|---|
| unseen | C1 | B9 | 4.072 | -2.509 | 11.14 | 3.086 | 4.724 | False |
| unseen | B9 | B1 | -1.192 | -10.27 | 4.67 | -2.121 | -0.232 | False |
| unseen | B3 | B1 | 5.401 | -6.518 | 22.14 | 4.413 | 6.061 | False |
| unseen | B7 | B1 | 1.914 | -2.993 | 6.263 | 1.427 | 2.251 | False |
| unseen | C1 | B1 | 2.88 | -3.699 | 7.107 | 2.283 | 3.277 | False |
| strict_draw | C1 | B9 | 1.062 | -4.289 | 6.239 | 0.09 | 1.813 | False |
| strict_draw | B7 | B1 | 0.678 | -1.668 | 4.291 | 0.407 | 0.937 | False |
| strict_draw | B3 | B1 | 2.698 | -1.124 | 11.53 | 1.605 | 3.443 | False |
| strict_draw | B9 | B1 | 0.588 | -4.248 | 4.933 | -0.552 | 1.581 | True |
| strict_draw | C1 | B1 | 1.65 | -0.798 | 5.175 | 1.161 | 1.891 | False |
| seen | C1 | B9 | 2.877 | 0.27 | 5.456 | 2.649 | 3.294 | False |
| seen | B7 | B1 | -0.489 | -1.583 | 0.615 | -0.553 | -0.308 | False |
| seen | B3 | B1 | 1.611 | -0.716 | 5.243 | 1.317 | 2.053 | False |
| seen | B9 | B1 | -2.944 | -5.403 | -0.309 | -3.336 | -2.619 | False |
| seen | C1 | B1 | -0.068 | -1.213 | 0.857 | -0.129 | 0.114 | True |


### U1' v2 -> v3 (same protocol, bigger dev region)

`results/u1p_v2v3.csv`

| method | mask | v2_pooled | v3_pooled | delta_v3_minus_v2 |
|---|---|---|---|---|
| B1 | seen | 7.269 | 10.19 | 2.924 |
| B1 | strict_draw | 15.23 | 14.45 | -0.779 |
| B1 | unseen | 12.53 | 11.96 | -0.561 |
| B2 | seen | 11.78 | nan | nan |
| B2 | strict_draw | 26.46 | nan | nan |
| B2 | unseen | 19.73 | nan | nan |
| B3 | seen | 7.453 | 11.8 | 4.351 |
| B3 | strict_draw | 16.4 | 17.15 | 0.747 |
| B3 | unseen | 13.51 | 17.36 | 3.859 |
| B7 | seen | 7.01 | 9.705 | 2.695 |
| B7 | strict_draw | 15.66 | 15.13 | -0.536 |
| B7 | unseen | 12.37 | 13.88 | 1.508 |
| B8 | seen | 14.22 | nan | nan |
| B8 | strict_draw | 26.94 | nan | nan |
| B8 | unseen | 22.07 | nan | nan |
| B9 | seen | 5.584 | 7.249 | 1.666 |
| B9 | strict_draw | 18.62 | 15.04 | -3.587 |
| B9 | unseen | 12.45 | 10.77 | -1.676 |
| C1 | seen | 7.447 | 10.13 | 2.679 |
| C1 | strict_draw | 16.23 | 16.1 | -0.132 |
| _(1 more rows in the CSV)_ | | | | |


Same v2 draws under both rules (`rules_disagree` = the old rule keeps 0 inside while (a) excludes it):

| mask | method_a | method_b | delta | old_lo | old_hi | a_lo | a_hi | rules_disagree |
|---|---|---|---|---|---|---|---|---|
| seen | B9 | B1 | -1.686 | -4.113 | 0.292 | -1.926 | -1.447 | True |
| seen | C1 | B1 | 0.178 | -0.445 | 0.991 | 0.105 | 0.254 | True |
| seen | B3 | B1 | 0.184 | -0.905 | 1.975 | 0.04 | 0.333 | True |
| seen | B7 | B1 | -0.26 | -1.077 | 0.497 | -0.345 | -0.169 | True |
| unseen | B9 | B1 | -0.076 | -8.954 | 12.83 | -1.057 | 0.939 | False |
| unseen | C1 | B1 | 1.43 | -0.628 | 4.932 | 1.187 | 1.723 | True |
| unseen | B3 | B1 | 0.981 | -4.169 | 6.558 | 0.416 | 1.525 | True |
| unseen | B7 | B1 | -0.155 | -2.462 | 2.655 | -0.412 | 0.113 | False |
| strict_draw | B9 | B1 | 3.396 | -6.702 | 18 | 2.095 | 4.854 | True |
| strict_draw | C1 | B1 | 1.003 | -0.689 | 5.586 | 0.717 | 1.327 | True |
| strict_draw | B3 | B1 | 1.172 | -3.268 | 9.279 | 0.52 | 1.879 | True |
| strict_draw | B7 | B1 | 0.436 | -1.836 | 5.085 | 0.1 | 0.786 | True |


### U8 ESN - pooled EER by enrolment size m (all clips)

`results/u8_esn.csv`

| mask | method | 0 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| seen | B1 | 10.19 | 10.22 | 10.11 | 10.06 |
| seen | B1+enroll | 10.19 | 9.63 | 9.707 | 9.257 |
| seen | ESN | 10.19 | 8.751 | 8.626 | 8.591 |
| seen | ESN-P | 10.19 | 8.749 | 8.576 | 8.497 |
| seen | OC | nan | 18.08 | 15.59 | 12.81 |
| seen | OC+B1 | nan | 7.996 | 6.388 | 4.397 |
| strict_draw | B1 | 14.45 | 14.31 | 14.32 | 14.3 |
| strict_draw | B1+enroll | 14.45 | 13.06 | 12.79 | 12.34 |
| strict_draw | ESN | 14.45 | 13.22 | 13.27 | 13.14 |
| strict_draw | ESN-P | 14.45 | 13.22 | 13.23 | 13 |
| strict_draw | OC | nan | 19.37 | 16.64 | 13.6 |
| strict_draw | OC+B1 | nan | 9.83 | 7.757 | 5.697 |
| unseen | B1 | 11.96 | 12.06 | 10.91 | 10.51 |
| unseen | B1+enroll | 11.96 | 10.92 | 9.829 | 9.153 |
| unseen | ESN | 11.96 | 10.5 | 9.591 | 9.235 |
| unseen | ESN-P | 11.96 | 10.49 | 9.523 | 9.113 |
| unseen | OC | nan | 24.59 | 21.16 | 17.98 |
| unseen | OC+B1 | nan | 11.84 | 8.737 | 6.194 |


U8 intervals (delta = method - control; negative = method better)

| m | mask | method_a | method_b | delta | old_lo | old_hi | b_lo | b_hi | meets_2pt_rule |
|---|---|---|---|---|---|---|---|---|---|
| 0 | unseen | ESN-P | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 0 | unseen | ESN-P | B1+enroll | 0 | 0 | 0 | 0 | 0 | False |
| 0 | unseen | B1+enroll | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 0 | unseen | ESN | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 0 | unseen | ESN | B1+enroll | 0 | 0 | 0 | 0 | 0 | False |
| 3 | unseen | ESN | B1 | -1.557 | -8.123 | 1.763 | -2.021 | -1.054 | False |
| 3 | unseen | ESN | B1+enroll | -0.423 | -4.206 | 2.241 | -0.511 | 0.139 | False |
| 3 | unseen | ESN-P | B1 | -1.561 | -8.123 | 1.82 | -2.018 | -1.064 | False |
| 0 | strict_draw | ESN | B1+enroll | 0 | 0 | 0 | 0 | 0 | False |
| 3 | unseen | ESN-P | B1+enroll | -0.427 | -4.112 | 2.34 | -0.513 | 0.129 | False |
| 3 | unseen | ESN | OC+B1 | -1.339 | -7.694 | 9.166 | -2.084 | -0.751 | False |
| 0 | strict_draw | ESN-P | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 0 | strict_draw | B1+enroll | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 0 | strict_draw | ESN-P | B1+enroll | 0 | 0 | 0 | 0 | 0 | False |
| 0 | strict_draw | ESN | B1 | 0 | 0 | 0 | 0 | 0 | False |
| 3 | unseen | ESN-P | OC+B1 | -1.343 | -7.742 | 9.285 | -2.1 | -0.758 | False |
| _(80 more rows in the CSV)_ | | | | | | | | | |


### U9 projection - packaged view

### U5 label-noise robustness

### Second backbone + Track Z (W17 / W8)

`results/w6_summary_v3_mms1b.csv` (mms-1b/S3 on the v3 dev region)

| method | clips | strict |
|---|---|---|
| B1 | 9.634 | 12.24 |
| B2 | 14.67 | 18.25 |
| B3 | 11.36 | 15.25 |
| B5 | 11.42 | 12.61 |
| B6 | 33.12 | 26.02 |
| B7 | 9.532 | 12.53 |
| B9 | 7.918 | 13.8 |
| C1 | 10.13 | 13.58 |


Track Z (k=0, v3 dev): `results/w8_zlabel.csv`

| method | pooled_eer | within_macro_eer | worst_platform_eer | auc |
|---|---|---|---|---|
| Z0 | 20.86 | 21.55 | 30.03 | 0.857 |
| Z1 | 25.62 | 21.55 | 30.03 | 0.813 |
| Z2 | 22.18 | 23.33 | 33.25 | 0.837 |
| Z5 | 16.16 | 18.41 | 36.73 | 0.894 |
| Z5-z1 | 23.74 | 21.04 | 31.02 | 0.851 |


`results/u5_noise.csv` (pooled EER, clips mask)

| method | k | 0.0 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| B1 | 30 | 16.86 | 18.49 | 21.13 | 28.84 |
| B1 | 100 | 13.52 | 14.8 | 16.17 | 19.25 |
| B1 | 290 | 10.95 | 12.48 | 13.07 | 15.61 |
| B9 | 30 | 17.47 | 19.5 | 22.14 | 33.32 |
| B9 | 100 | 11.66 | 14.73 | 16.58 | 22.37 |
| B9 | 290 | 8.074 | 9.749 | 11.97 | 17.14 |
| C1 | 30 | 17.55 | 19.74 | 23.11 | 32.1 |
| C1 | 100 | 13.95 | 16.69 | 20.56 | 26.14 |
| C1 | 290 | 11.4 | 16.09 | 20.71 | 29.19 |
| M-ALP | 30 | 16.17 | 17.86 | 20.57 | 28.77 |
| M-ALP | 100 | 12.14 | 13.47 | 14.99 | 19 |
| M-ALP | 290 | 8.868 | 9.87 | 11.2 | 14.61 |


`results/u9_projection.csv` (CV-selected variant per draw)

| method | seen | strict_draw | unseen |
|---|---|---|---|
| B1 | 10.19 | 14.45 | 11.96 |
| GRID-ind | nan | 14.47 | nan |
| GRID-tra | nan | 14.39 | nan |
| PROJ-ind | 10.26 | 14.47 | 12.04 |
| PROJ-tra | 10.2 | 14.48 | 11.79 |


### G17 (N5, ESN)

```json
{
  "b_excludes_zero": true,
  "best_control": "OC+B1",
  "delta_points": 5.508007225318261,
  "effect_ge_2": false,
  "interval_b": [
    4.231945025882896,
    6.794665527272493
  ],
  "interval_old": [
    1.5942066744745678,
    10.43977702993291
  ],
  "m": 5,
  "mask": "strict_draw",
  "means": {
    "B1": 14.320741630485031,
    "B1+enroll": 12.790540481510225,
    "ESN": 13.265034648768358,
    "ESN-P": 13.229142474350276,
    "OC": 16.641382975051055,
    "OC+B1": 7.757027423450097
  },
  "population": "all",
  "verdict": "ESN_FAIL"
}
```

### G18 (N6, projection)

```json
{
  "b1_strict_draw": 14.44969564859901,
  "b_excludes_zero": false,
  "best_grid_cells": [
    {
      "m_p": 0,
      "m_s": 4,
      "pooled_eer": 14.195042117053617,
      "scope": "transductive"
    },
    {
      "m_p": 2,
      "m_s": 8,
      "pooled_eer": 14.246976912704993,
      "scope": "transductive"
    },
    {
      "m_p": 0,
      "m_s": 8,
      "pooled_eer": 14.262488833280134,
      "scope": "transductive"
    },
    {
      "m_p": 2,
      "m_s": 32,
      "pooled_eer": 14.370609707824519,
      "scope": "transductive"
    },
    {
      "m_p": 0,
      "m_s": 32,
      "pooled_eer": 14.378213044843621,
      "scope": "transductive"
    }
  ],
  "best_grid_note": "choosing the best cell after seeing strict_draw is optimistic; the CV-selected number above is the pre-registered comparison",
  "cv_selected": {
    "seen|B1": 10.193662061010695,
    "seen|PROJ-ind": 10.264264824365384,
    "seen|PROJ-tra": 10.197554465065478,
    "strict_draw|B1": 14.44969564859901,
    "strict_draw|GRID-ind": 14.466019411372853,
    "strict_draw|GRID-tra": 14.387026227125485,
    "strict_draw|PROJ-ind": 14.467374819087011,
    "strict_draw|PROJ-tra": 14.475845254680719,
    "unseen|B1": 11.963969067064927,
    "unseen|PROJ-ind": 12.044137242407123,
    "unseen|PROJ-tra": 11.78452988987242
  },
  "delta_vs_b1": 0.017679170488001716,
  "effect_ge_2": false,
  "interval_b": [
    -0.018825364972594124,
    0.06187870655338064
  ],
  "method": "PROJ-ind",
  "seen": 10.264264824365384,
  "seen_delta": 0.07060276335468885,
  "seen_within_tolerance": true,
  "strict_draw": 14.467374819087011,
  "verdict": "PROJ_FAIL"
}
```

### G12-G16 (U4 / M-ALP)

```json
{
  "G12": {
    "b1_unseen_reference": 11.96,
    "best_selector_lr_lp_strict_draw_k290": 12.469497909007828,
    "rule": "N1' is informational after \u8ffd\u52a0 02; the 0.5-point rule is no longer a novelty gate",
    "status": "MALP_INFO"
  },
  "G13": {
    "best_other": "E",
    "gain_vs_best_other": null,
    "gain_vs_uniform": 1.2946247304802752,
    "interval_vs_best_other": null,
    "interval_vs_uniform": [
      -1.7600999721144461,
      -0.7091938393260055
    ],
    "k": 100,
    "malp": 13.66069160340539,
    "mask": "strict_draw",
    "others": {
      "E": 13.277315350923827,
      "K": 14.70625124378702,
      "TypiClust": 13.899041337614628
    },
    "status": "SEL_FAIL",
    "uniform": 14.955316333885664
  },
  "G14": {
    "cc": 13.449528021703127,
    "delta_sc_minus_cc": 0.5806852138160487,
    "interval": [
      0.07186872520276137,
      1.4024332409723443
    ],
    "sc": 14.030213235519177,
    "status": "SPEAKER_COVERAGE_FAIL"
  },
  "G15": {
    "b1u_k290": 14.88380972129316,
    "difference": -1.2231181178877701,
    "malp_k100": 13.66069160340539,
    "status": "LABEL_EFFICIENCY_PASS"
  },
  "G16": {
    "baselines": {
      "E": 12.534758324422466,
      "K": 14.02398558420758,
      "TypiClust": 13.501778737556043,
      "U": 14.88380972129316
    },
    "beats": {
      "E": false,
      "K": true,
      "TypiClust": true,
      "U": true
    },
    "malp_k290": 13.079254037337861,
    "reason": "S-F' needs every BASE_F' baseline on both masks and two backbones; this run covers the selector comparators on one backbone, so the gate stays negative by rule",
    "status": "SOTA_F2_DEV_NO"
  }
}
```

### W15 ITW availability

```json
{
  "local_dataset": null,
  "probes": {
    "huggingface": {
      "reachable": false,
      "reason": "HTTP 401 Unauthorized",
      "status": 401,
      "url": "https://huggingface.co/api/models/mueller91/In-The-Wild"
    },
    "mirror": {
      "reachable": false,
      "reason": "HTTP 401 Unauthorized",
      "status": 401,
      "url": "https://hf-mirror.com/api/models/mueller91/In-The-Wild"
    }
  },
  "reason": "the dataset is not on the box and the public endpoints answer huggingface=401, mirror=401; TASKBOOK 1.2 says gated=false",
  "taskbook_claim": "gated=false (TASKBOOK 1.2)",
  "verdict": "ITW_UNAVAILABLE"
}
```

### W21 ADD-GP attempt

```json
{
  "clone": {
    "cloned": true,
    "path": "/root/autodl-tmp/data/x3/vendor/ADD-GP",
    "skipped": true
  },
  "readme_head": "# ADD-GP\n\n### Official implementation for Few-Shot Speech Deepfake Detection Adaptation with Gaussian Processes\n#### Interspeech 2025\nRecent advancements in Text-to-Speech (TTS) models, particularly in voice cloning, have intensified the demand for adaptable and efficient deepfake detection methods. As TTS systems continue to evolve, detection models must be able to efficiently adapt to previously unseen generation models with minimal data. This paper introduces ADD-GP, a few-shot adaptive framework based on a Gaussian Process (GP) classifier for Audio Deepfake Detection (ADD). We show how the",
  "reason": "no recognisable entry point (looked for main.py, train.py, run.py, eval.py, addgp.py, demo.py, inference.py); the checkout holds 9 files and 7 python files: code/models/GP.py, code/models/__init__.py, code/tts_trainer.py, code/utils/__init__.py, code/utils/tts_dataset.py, code/utils/tts_personalized_dataset.py; no weights either",
  "repo": {
    "entry_points": [],
    "exists": true,
    "has_readme": true,
    "n_files": 9,
    "python_files": [
      "code/models/GP.py",
      "code/models/__init__.py",
      "code/tts_trainer.py",
      "code/utils/__init__.py",
      "code/utils/tts_dataset.py",
      "code/utils/tts_personalized_dataset.py",
      "code/utils/utils.py"
    ],
    "requirements": [
      "requirements.txt: # This file may be used to create an environment using: | # $ conda create --name <env> --file <this file> | # platform: linux-64 | _libgcc_mutex=0.1=main | _openmp_mutex=5.1=1_gnu | absl-py=2.1.0=pypi_0 | accelerate=0.31.0=pypi_0 | aiofiles=23.2.1=pypi_0 | aiohttp=3.9.5=pypi_0 | aiosignal=1.3.1=pypi_0 | altair=5.3.0=pypi_0 | annotated-types=0.7.0=pypi_0 | anyascii=0.3.2=pypi_0 | anyio=4.4.0=pypi_"
    ],
    "weights": []
  },
  "verdict": "ADDGP_UNAVAILABLE"
}
```

### U1' verdicts

```json
{
  "B9_minus_B1_unseen": -1.1918230881230043,
  "B9_minus_B1_unseen_ci_b": [
    -2.1206788684265963,
    -0.2320875955407337
  ],
  "B9_minus_B1_unseen_ci_old": [
    -10.264579860558005,
    4.670090701425648
  ],
  "G4_C1_minus_B1_strict_draw": {
    "C1_beats_B1": false,
    "b": [
      1.161454315123263,
      1.8905929410866749
    ],
    "delta": 1.650156500202433,
    "old": [
      -0.7981131240316968,
      5.175322309599283
    ]
  },
  "LP_GAIN_IS_SPEAKER_LOCAL": false,
  "LP_GAIN_UNSEEN_BELOW_RESOLUTION": true,
  "LP_GAIN_UNSEEN_MEETS_2PT_BAR": false,
  "SPEAKER_GAP": false,
  "SPEAKER_GAP_methods": [],
  "k": 290,
  "speaker_ratio": {
    "B1": {
      "ratio": 1.1736674215270884,
      "seen": 10.193662061010695,
      "unseen": 11.963969067064927
    },
    "B3": {
      "ratio": 1.4710586877781557,
      "seen": 11.804579056338861,
      "unseen": 17.365228576391345
    },
    "B7": {
      "ratio": 1.4300157780168998,
      "seen": 9.704528843502585,
      "unseen": 13.877629364428794
    },
    "B9": {
      "ratio": 1.485933688556856,
      "seen": 7.249412313549381,
      "unseen": 10.772145978941921
    },
    "C1": {
      "ratio": 1.4659161284098254,
      "seen": 10.12611518787006,
      "unseen": 14.84403557203441
    }
  }
}
```

- {'item': 'TASKBOOK 1.2 headline attribution', 'what': "TASKBOOK says Line A's headline 13.64/14.56/20.79 comes from 'frozen XLS-R features + a small probe trained on 290 labels, plus cell-prior correction'. Measured on the box, that route gives 24.65 pooled (k=10, BB+EB), and even an ORACLE cell prior only reaches 21.52 pooled on the p0cal scores. The 13.64 number is COMPARISON.md's line A: 290 labels train a head directly with NO prior correction (clips 13.05 / strict 13.64, AUC 0.94).", 'impact': "S-F must be judged against the ADAPTATION route (BASE_F's B11 family), not only against the prior-correction family; otherwise the SOTA bar is ~9 EER points too low. The prior-correction family stays in BASE_F under its own name.", 'source': 'W4 on the box, 2026-10-14; wx2/docs/COMPARISON.md 0.0 line 42'}
- {'item': 'B10_REDUCED is not a usable baseline', 'what': "B10_REDUCED scored mean AUC 0.412 (< 0.5) over 50 draws: the head never trained. With a random initialisation the declared budget (5 epochs, lr <= 3e-5, AdamW) leaves the weights essentially at their random values, so the score is a random projection dominated by PC1 and comes out anti-correlated with the label. The full B10 (W13 on the GPU lane) is supposed to start from the backbone's own pretrained head, which this variant cannot do.", 'impact': "Any 'we beat B10' statement made with these rows is vacuous. Either run the full B10 on the encoder, or report B10 as NOT_MEANINGFUL and keep it out of the S-F comparison.", 'source': 'W13 on the box, 2026-10-14'}
- {'item': 'W6 B6 detector was wired to Line A for every backbone', 'what': 'scripts/w6_baselines.py fed B6 the frozen Line A probe score (last_scores_p0cal_raw_s0.npy) regardless of --backbone, so B6 was identical on mms-1b and xls-r-2b (33.12 clips / 26.02 strict) and much worse than the true zero-shot (20.86). With --b6-zero backbone the same draws give xls-r-2b 18.92/19.94 and mms-1b 23.32/23.22.', 'impact': 'No S-F claim used B6 (it is not in BASE_F prime), but the stored B6 rows overstate its error by 6 to 14 points and hid the backbone dependence. The default stays lineA so old files remain reproducible; new runs should pass --b6-zero backbone.', 'source': 'w6b/w6a v3 reruns on the box, 2026-10-15 (results/w6_summary_v3_*_b6fix.csv)'}

## W2 zero-shot

Per-backbone zero-shot three views.

| pooled_eer | within_macro_eer | worst_platform | worst_platform_eer | auc | hter_pooled | n_eval | n_pos | eer_bilibili | eer_douyin | eer_facebook | eer_rednote | eer_tiktok | eer_x | eer_youtube | backbone | kind | fidelity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20.21 | 21.52 | tiktok | 23.89 | 0.8627 | 35.6 | 24529 | 6361 | 21.91 | 19.23 | 22.5 | 23.62 | 23.89 | 18.22 | 21.27 | xls-r-2b | antideepfake | pass |
| 22.62 | 22.04 | rednote | 27.62 | 0.8501 | 35.73 | 24529 | 6361 | 24.59 | 23.2 | 20.91 | 27.62 | 20.33 | 19.33 | 18.28 | xls-r-1b | antideepfake | pass |
| 26.32 | 23.59 | douyin | 29.76 | 0.82 | 39.62 | 24529 | 6361 | 23.08 | 29.76 | 20.47 | 29.57 | 22.99 | 20.28 | 19 | mms-1b | antideepfake | pass |
| 32.26 | 50 | bilibili | 50 | 0.7531 | 34.19 | 24529 | 6361 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | metadata-lookup | metadata | nan |


## W4 reconciliation

3-draw headline vs 100-draw mean.

| scope | metric | headline | rerun | diff | mask | n_draws | reference |
|---|---|---|---|---|---|---|---|
| 3draw | pooled_eer | 24.65 | 25.85 | 1.198 | clips | 3 | fsc_var_results.csv (bb, k=10) |
| 3draw | within_macro_eer | 34.36 | 35.07 | 0.7102 | clips | 3 | fsc_var_results.csv (bb, k=10) |
| 100draw | pooled_eer | 24.65 | 25.1 | 0.4479 | clips | 100 | fsc_var_results.csv (bb, k=10) |
| 100draw | within_macro_eer | 34.36 | 34.24 | -0.1162 | clips | 100 | fsc_var_results.csv (bb, k=10) |


## W4 mechanism

M0 / M1 / M2 under one draw protocol.

| method | mask | pooled_eer_mean | pooled_eer_std | within_macro_eer_mean | within_macro_eer_std | worst_platform_eer_mean | worst_platform_eer_std | auc_mean | auc_std |
|---|---|---|---|---|---|---|---|---|---|
| M0 | clips | 32.43 | 1.128 | 36.54 | 1.533 | 44.65 | 1.328 | 0.7469 | 0.01585 |
| M0 | strict | 30.02 | 2.567 | 35.81 | 3.776 | 50.97 | 8.038 | 0.7706 | 0.03263 |
| M1 | clips | 30.36 | 2.27 | 44.4 | 1.442 | 52.06 | 3.233 | 0.7627 | 0.01505 |
| M1 | strict | 25.77 | 3.856 | 50.37 | 3.747 | 64.61 | 15.27 | 0.7919 | 0.03611 |
| M2 | clips | 25.1 | 1.494 | 34.24 | 1.771 | 44.65 | 1.328 | 0.8281 | 0.01316 |
| M2 | strict | 22.63 | 2.395 | 37.25 | 4.193 | 56.02 | 13.25 | 0.8474 | 0.02415 |


## W5 backbone rank

Candidate (backbone, spec) ranking.

| backbone | spec | metric | mean | sd | paired_mean | paired_lo | paired_hi | paired_n | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| xls-r-2b | S1 | pooled_eer | 12.38 | 2.78 | -7.38 | -11.83 | -3.244 | 30 | 30 |
| xls-r-2b | S1 | within_macro_eer | 10.38 | 1.403 | -5.661 | -9.261 | -2.909 | 30 | 30 |
| xls-r-2b | S1 | worst_platform_eer | 18.04 | 7.1 | -9.001 | -21.72 | 4.008 | 30 | 30 |
| xls-r-2b | S1 | auc | 0.9432 | 0.01737 | nan | nan | nan | 0 | 30 |
| xls-r-2b | S2 | pooled_eer | 12.05 | 2.225 | -7.708 | -13.3 | -2.489 | 30 | 30 |
| xls-r-2b | S2 | within_macro_eer | 11.18 | 1.376 | -4.859 | -8.724 | -1.767 | 30 | 30 |
| xls-r-2b | S2 | worst_platform_eer | 19.13 | 5.209 | -7.911 | -25.7 | 5.939 | 30 | 30 |
| xls-r-2b | S2 | auc | 0.9455 | 0.01498 | nan | nan | nan | 0 | 30 |
| xls-r-2b | S3 | pooled_eer | 12.17 | 2.984 | -7.586 | -13.43 | -3.735 | 30 | 30 |
| xls-r-2b | S3 | within_macro_eer | 10.25 | 1.168 | -5.795 | -9.601 | -2.634 | 30 | 30 |
| xls-r-2b | S3 | worst_platform_eer | 17.4 | 5.358 | -9.643 | -25.58 | 2.007 | 30 | 30 |
| xls-r-2b | S3 | auc | 0.945 | 0.01684 | nan | nan | nan | 0 | 30 |
| xls-r-2b | S4 | pooled_eer | 12.08 | 2.164 | -7.682 | -12.79 | -2.889 | 30 | 30 |
| xls-r-2b | S4 | within_macro_eer | 11.02 | 1.379 | -5.019 | -8.636 | -1.862 | 30 | 30 |
| xls-r-2b | S4 | worst_platform_eer | 18.93 | 5.719 | -8.114 | -24.34 | 4.181 | 30 | 30 |
| xls-r-2b | S4 | auc | 0.9456 | 0.01455 | nan | nan | nan | 0 | 30 |
| xls-r-2b | S5 | pooled_eer | 12.17 | 2.979 | -7.592 | -13.45 | -3.735 | 30 | 30 |
| xls-r-2b | S5 | within_macro_eer | 10.25 | 1.165 | -5.793 | -9.6 | -2.642 | 30 | 30 |
| xls-r-2b | S5 | worst_platform_eer | 17.4 | 5.36 | -9.639 | -25.58 | 2.007 | 30 | 30 |
| xls-r-2b | S5 | auc | 0.945 | 0.01684 | nan | nan | nan | 0 | 30 |
| xlsr300m-linea | S1-lineA | pooled_eer | 19.76 | 3.774 | nan | nan | nan | 0 | 30 |
| xlsr300m-linea | S1-lineA | within_macro_eer | 16.04 | 2.314 | nan | nan | nan | 0 | 30 |
| xlsr300m-linea | S1-lineA | worst_platform_eer | 27.04 | 9.389 | nan | nan | nan | 0 | 30 |
| xlsr300m-linea | S1-lineA | auc | 0.895 | 0.02566 | nan | nan | nan | 0 | 30 |


## W6 summary

k=290, every baseline plus C1.

| backbone | spec | method | mask | metric | mean | std |
|---|---|---|---|---|---|---|
| xls-r-2b | S3 | B1 | clips | pooled_eer | 7.695 | 1.093 |
| xls-r-2b | S3 | B1 | clips | within_macro_eer | 8.763 | 1.311 |
| xls-r-2b | S3 | B1 | clips | worst_platform_eer | 16.13 | 5.204 |
| xls-r-2b | S3 | B1 | clips | auc | 0.9731 | 0.004672 |
| xls-r-2b | S3 | B1 | strict | pooled_eer | 12.52 | 4.199 |
| xls-r-2b | S3 | B1 | strict | within_macro_eer | 11.16 | 5.829 |
| xls-r-2b | S3 | B1 | strict | worst_platform_eer | 12.29 | 7.121 |
| xls-r-2b | S3 | B1 | strict | auc | 0.9378 | 0.02862 |
| xls-r-2b | S3 | B11 | clips | pooled_eer | 26.62 | 1.875 |
| xls-r-2b | S3 | B11 | clips | within_macro_eer | 26.22 | 1.704 |
| xls-r-2b | S3 | B11 | clips | worst_platform_eer | 42.26 | 5.414 |
| xls-r-2b | S3 | B11 | clips | auc | 0.8073 | 0.0208 |
| xls-r-2b | S3 | B11 | strict | pooled_eer | 35.15 | 6.137 |
| xls-r-2b | S3 | B11 | strict | within_macro_eer | 36.74 | 7.66 |
| xls-r-2b | S3 | B11 | strict | worst_platform_eer | 38.16 | 8.475 |
| xls-r-2b | S3 | B11 | strict | auc | 0.7005 | 0.07392 |
| xls-r-2b | S3 | B2 | clips | pooled_eer | 12.55 | 2.966 |
| xls-r-2b | S3 | B2 | clips | within_macro_eer | 10.44 | 1.485 |
| xls-r-2b | S3 | B2 | clips | worst_platform_eer | 17.86 | 5.262 |
| xls-r-2b | S3 | B2 | clips | auc | 0.942 | 0.01681 |
| xls-r-2b | S3 | B2 | strict | pooled_eer | 19.73 | 6.993 |
| xls-r-2b | S3 | B2 | strict | within_macro_eer | 11.98 | 6.215 |
| xls-r-2b | S3 | B2 | strict | worst_platform_eer | 13.36 | 8.454 |
| xls-r-2b | S3 | B2 | strict | auc | 0.8625 | 0.06953 |
| xls-r-2b | S3 | B3 | clips | pooled_eer | 7.916 | 1.488 |
| xls-r-2b | S3 | B3 | clips | within_macro_eer | 8.763 | 1.311 |
| xls-r-2b | S3 | B3 | clips | worst_platform_eer | 16.13 | 5.204 |
| xls-r-2b | S3 | B3 | clips | auc | 0.9705 | 0.006698 |
| xls-r-2b | S3 | B3 | strict | pooled_eer | 13.51 | 4.897 |
| xls-r-2b | S3 | B3 | strict | within_macro_eer | 11.16 | 5.829 |
| xls-r-2b | S3 | B3 | strict | worst_platform_eer | 12.29 | 7.121 |
| xls-r-2b | S3 | B3 | strict | auc | 0.9284 | 0.03916 |
| xls-r-2b | S3 | B5 | clips | pooled_eer | 9.067 | 1.028 |
| xls-r-2b | S3 | B5 | clips | within_macro_eer | 9.372 | 0.8159 |
| xls-r-2b | S3 | B5 | clips | worst_platform_eer | 15.79 | 3.218 |
| xls-r-2b | S3 | B5 | clips | auc | 0.969 | 0.00417 |
| xls-r-2b | S3 | B5 | strict | pooled_eer | 14.03 | 4.819 |
| xls-r-2b | S3 | B5 | strict | within_macro_eer | 12.51 | 5.92 |
| xls-r-2b | S3 | B5 | strict | worst_platform_eer | 13.69 | 6.671 |
| xls-r-2b | S3 | B5 | strict | auc | 0.9324 | 0.03086 |
| _(32 more rows in the CSV)_ | | | | | | |


## W6 paired

C1 vs B1/B2/B3/B11, draw-paired.

| method | other | metric | mean | lo | hi | n | backbone | spec |
|---|---|---|---|---|---|---|---|---|
| C1 | B1 | pooled_eer | 0.3031 | -0.3762 | 1.163 | 100 | xls-r-2b | S3 |
| C1 | B1 | within_macro_eer | 0.4325 | -0.5027 | 1.522 | 100 | xls-r-2b | S3 |
| C1 | B1 | worst_platform_eer | 0.3744 | -3.347 | 3.388 | 100 | xls-r-2b | S3 |
| C1 | B2 | pooled_eer | -4.556 | -10.06 | 0.2749 | 100 | xls-r-2b | S3 |
| C1 | B2 | within_macro_eer | -1.241 | -3.382 | 0.5556 | 100 | xls-r-2b | S3 |
| C1 | B2 | worst_platform_eer | -1.348 | -10.92 | 3.306 | 100 | xls-r-2b | S3 |
| C1 | B3 | pooled_eer | 0.0818 | -2.019 | 1.678 | 100 | xls-r-2b | S3 |
| C1 | B3 | within_macro_eer | 0.4325 | -0.5027 | 1.522 | 100 | xls-r-2b | S3 |
| C1 | B3 | worst_platform_eer | 0.3744 | -3.347 | 3.388 | 100 | xls-r-2b | S3 |
| C1 | B11 | pooled_eer | -18.62 | -21.9 | -15.38 | 100 | xls-r-2b | S3 |
| C1 | B11 | within_macro_eer | -17.02 | -19.94 | -14.02 | 100 | xls-r-2b | S3 |
| C1 | B11 | worst_platform_eer | -25.75 | -38.93 | -12.91 | 100 | xls-r-2b | S3 |


## W7 budget curve

k in {30, 100, 290, 600, 1200}.

| method | k | pooled_eer_mean | pooled_eer_std | within_macro_eer_mean | within_macro_eer_std | worst_platform_eer_mean | worst_platform_eer_std | auc_mean | auc_std | n_labeled_mean | n_labeled_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 29 | 28.13 | 5.14 | 31.03 | 3.514 | 56.18 | 7.51 | 0.7899 | 0.05305 | 21 | 0 |
| B1 | 87 | 20.1 | 3.474 | 22.91 | 3.372 | 44.36 | 11.7 | 0.8764 | 0.0306 | 63 | 0 |
| B1 | 290 | 13.02 | 1.805 | 14.7 | 2.327 | 27.02 | 8.571 | 0.9379 | 0.01275 | 204 | 0 |
| B1 | 580 | 10.37 | 1.508 | 11.62 | 1.512 | 20.03 | 4.074 | 0.9564 | 0.009554 | 401 | 0 |
| B1 | 1160 | 8.094 | 1.037 | 9.396 | 0.907 | 16.77 | 2.838 | 0.9704 | 0.005342 | 724 | 0 |
| B11 | 29 | 41.78 | 1.334 | 42.58 | 1.596 | 51.64 | 4.005 | 0.6204 | 0.01734 | 21 | 0 |
| B11 | 87 | 35.53 | 1.851 | 36.69 | 2.192 | 47.17 | 5.027 | 0.7014 | 0.02268 | 63 | 0 |
| B11 | 290 | 25.56 | 1.779 | 26.67 | 1.757 | 38.86 | 4.79 | 0.8206 | 0.01861 | 204 | 0 |
| B11 | 580 | 19.12 | 1.617 | 20.15 | 1.431 | 30.44 | 5.181 | 0.8874 | 0.01483 | 401 | 0 |
| B11 | 1160 | 12.92 | 1.734 | 13.91 | 1.648 | 22.07 | 4.466 | 0.9408 | 0.01184 | 724 | 0 |
| B2 | 29 | 28.13 | 5.14 | 31.03 | 3.514 | 56.18 | 7.51 | 0.7899 | 0.05305 | 21 | 0 |
| B2 | 87 | 26.08 | 4.149 | 24.24 | 3.541 | 43.85 | 12.08 | 0.8254 | 0.03705 | 63 | 0 |
| B2 | 290 | 18.66 | 3.89 | 15.89 | 2.337 | 26.49 | 8.437 | 0.8998 | 0.02722 | 204 | 0 |
| B2 | 580 | 16.84 | 2.908 | 13.26 | 1.587 | 20.22 | 3.842 | 0.9141 | 0.01992 | 401 | 0 |
| B2 | 1160 | 14.77 | 2.554 | 11.28 | 0.8434 | 17.34 | 2.664 | 0.9315 | 0.01574 | 724 | 0 |
| B3 | 29 | 28.44 | 4.509 | 31.03 | 3.514 | 56.18 | 7.51 | 0.7895 | 0.04627 | 21 | 0 |
| B3 | 87 | 20.88 | 3.402 | 22.91 | 3.372 | 44.36 | 11.7 | 0.869 | 0.03138 | 63 | 0 |
| B3 | 290 | 13.83 | 2.315 | 14.7 | 2.327 | 27.02 | 8.571 | 0.9335 | 0.01626 | 204 | 0 |
| B3 | 580 | 11.34 | 2.058 | 11.62 | 1.512 | 20.03 | 4.074 | 0.9501 | 0.01395 | 401 | 0 |
| B3 | 1160 | 9.652 | 1.871 | 9.396 | 0.907 | 16.77 | 2.838 | 0.9624 | 0.01093 | 724 | 0 |
| C1 | 29 | 28.13 | 5.083 | 30.84 | 3.664 | 54.37 | 7.621 | 0.7888 | 0.05466 | 21 | 0 |
| C1 | 87 | 20.3 | 3.677 | 22.63 | 3.491 | 40.75 | 11.51 | 0.8737 | 0.03414 | 63 | 0 |
| C1 | 290 | 13.43 | 2.015 | 14.94 | 2.325 | 26.32 | 8.434 | 0.9348 | 0.01439 | 204 | 0 |
| C1 | 580 | 11.17 | 1.646 | 12.39 | 1.675 | 20.29 | 4.368 | 0.951 | 0.01164 | 401 | 0 |
| C1 | 1160 | 9.132 | 1.289 | 10.45 | 1.183 | 17.75 | 3.719 | 0.9646 | 0.00776 | 724 | 0 |
| PAST | 29 | 27.58 | 5.204 | 31.47 | 3.815 | 59.8 | 7.907 | 0.7896 | 0.0559 | 21 | 0 |
| PAST | 87 | 20.03 | 3.814 | 23.2 | 3.867 | 46.11 | 13.93 | 0.8716 | 0.03638 | 63 | 0 |
| PAST | 290 | 12.75 | 1.972 | 14.33 | 2.506 | 25.62 | 9.671 | 0.9388 | 0.01461 | 204 | 0 |
| PAST | 580 | 10.76 | 1.586 | 12.07 | 1.598 | 19.85 | 3.975 | 0.9543 | 0.01093 | 401 | 0 |
| PAST | 1160 | 8.969 | 1.243 | 10.28 | 1.099 | 17.63 | 3.558 | 0.9662 | 0.007243 | 724 | 0 |


## W8 Track Z

Z0 - Z3, zero labels.

| pooled_eer | within_macro_eer | worst_platform | worst_platform_eer | auc | hter_pooled | n_eval | n_pos | eer_bilibili | eer_douyin | eer_facebook | eer_rednote | eer_tiktok | eer_x | eer_youtube | protocol | region | backbone | spec | method | k | mask | n_labeled | mu0 | mu1 | sig0 | sig1 | pi_global | loglik | seed_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20.86 | 21.54 | tiktok | 30.03 | 0.8573 | nan | 15002 | 3990 | 17.2 | 18.34 | 25.2 | 21.42 | 30.03 | 17.84 | 20.8 | track_z_v3_dev | dev | xls-r-2b | S3 | Z0 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | nan |
| 25.62 | 21.54 | tiktok | 30.03 | 0.8127 | nan | 15002 | 3990 | 17.2 | 18.34 | 25.2 | 21.42 | 30.03 | 17.84 | 20.8 | track_z_v3_dev | dev | xls-r-2b | S3 | Z1 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | nan |
| 22.18 | 23.33 | tiktok | 33.25 | 0.8373 | nan | 15002 | 3990 | 17.2 | 18.34 | 27.17 | 26.24 | 33.25 | 18.04 | 23.08 | track_z_v3_dev | dev | xls-r-2b | S3 | Z2 | 0 | clips | 0 | -105.3 | 0.3661 | 12.2 | 74.7 | 0.2642 | -4.777 | nan |
| 16.16 | 18.41 | tiktok | 36.73 | 0.8941 | nan | 15002 | 3990 | 17.14 | 10.92 | 13.57 | 20.07 | 36.73 | 14.56 | 15.9 | track_z_v3_dev | dev | xls-r-2b | S3 | Z5 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | Z2 |
| 23.74 | 21.04 | tiktok | 31.02 | 0.8508 | nan | 15002 | 3990 | 20.8 | 14.82 | 29.57 | 15.66 | 31.02 | 16.17 | 19.19 | track_z_v3_dev | dev | xls-r-2b | S3 | Z5-z1 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | Z1 |


## W9 audit

Leakage and negative controls.

| check | method | mask | eer | reference_eer | n_draws | flag | expectation |
|---|---|---|---|---|---|---|---|
| label-permutation-global | B1 | clips | 50.3 | 50 | 30 | False | fully shuffled labels -> EER ~ 50 (+-3) |
| label-permutation-within-platform | B1 | clips | 44.38 | 30.59 | 30 | False | the platform mix is preserved, so the correct floor is the metadata-only level; beating it by >3 means leakage |
| cell-permutation | B1 | clips | 10.8 | 13.28 | 30 | False | C1's gain over the permuted-cell run disappears |
| clips-vs-strict-baseline | B1 | clips | 13.28 | nan | 30 | False | baseline (no expectation) |
| label-permutation-global | B2 | clips | 50.12 | 50 | 30 | False | fully shuffled labels -> EER ~ 50 (+-3) |
| label-permutation-within-platform | B2 | clips | 40.45 | 30.59 | 30 | False | the platform mix is preserved, so the correct floor is the metadata-only level; beating it by >3 means leakage |
| cell-permutation | B2 | clips | 12.23 | 19.76 | 30 | False | C1's gain over the permuted-cell run disappears |
| clips-vs-strict-baseline | B2 | clips | 19.76 | nan | 30 | False | baseline (no expectation) |
| label-permutation-global | C1 | clips | 50.21 | 50 | 30 | False | fully shuffled labels -> EER ~ 50 (+-3) |
| label-permutation-within-platform | C1 | clips | 45.97 | 30.59 | 30 | False | the platform mix is preserved, so the correct floor is the metadata-only level; beating it by >3 means leakage |
| cell-permutation | C1 | clips | 10.79 | 13.67 | 30 | False | C1's gain over the permuted-cell run disappears |
| clips-vs-strict-baseline | C1 | clips | 13.67 | nan | 30 | False | baseline (no expectation) |
| label-permutation-global | B7 | clips | 50.19 | 50 | 30 | False | fully shuffled labels -> EER ~ 50 (+-3) |
| label-permutation-within-platform | B7 | clips | 45.6 | 30.59 | 30 | False | the platform mix is preserved, so the correct floor is the metadata-only level; beating it by >3 means leakage |
| cell-permutation | B7 | clips | 10.28 | 12.67 | 30 | False | C1's gain over the permuted-cell run disappears |
| clips-vs-strict-baseline | B7 | clips | 12.67 | nan | 30 | False | baseline (no expectation) |
| near-duplicate-share | - | clips | 0.0001936 | nan | 30 | False | share of eval clips with a labelled neighbour > cos 0.99 below 5% |
| metadata-only | platform x language | clips | 30.59 | nan | 1 | False | shortcut level; any audio claim must beat this |


## W11 PAST

PAST and its ablations.

| method | other | metric | mean | lo | hi | n | backbone | spec | views |
|---|---|---|---|---|---|---|---|---|---|
| PAST | C1 | pooled_eer | -0.2249 | -1.754 | 0.981 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | C1 | within_macro_eer | 0.2783 | -0.9491 | 1.879 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | B7 | pooled_eer | 0.2843 | -0.8806 | 1.337 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | B7 | within_macro_eer | 0.6188 | -0.5258 | 1.869 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-noEM | pooled_eer | 0.1414 | -0.3853 | 0.7253 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-noEM | within_macro_eer | 0.07589 | -0.3685 | 0.6376 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-noQuota | pooled_eer | -0.2499 | -0.7874 | 0.3539 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-noQuota | within_macro_eer | 0.02246 | -0.5179 | 0.9622 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-1round | pooled_eer | 0.301 | -0.3907 | 1.005 | 50 | xls-r-2b | S3 | famA1,famA2 |
| PAST | PAST-1round | within_macro_eer | 0.5097 | -0.2986 | 1.558 | 50 | xls-r-2b | S3 | famA1,famA2 |


## W12 CASA

Selection strategies at k=100/290.

| method | other | metric | k | mean | lo | hi | n | backbone | spec |
|---|---|---|---|---|---|---|---|---|---|
| CASA | U | pooled_eer | 100 | -6.416 | -12.05 | -0.8107 | 30 | xlsr300m-linea | S1-lineA |
| CASA | U | within_macro_eer | 100 | -5.008 | -10.74 | 1.443 | 30 | xlsr300m-linea | S1-lineA |
| CASA | P | pooled_eer | 100 | -2.851 | -6.983 | 1.821 | 30 | xlsr300m-linea | S1-lineA |
| CASA | P | within_macro_eer | 100 | -3.398 | -11.37 | 2.024 | 30 | xlsr300m-linea | S1-lineA |
| CASA | N | pooled_eer | 100 | -2.849 | -7.109 | 1.821 | 30 | xlsr300m-linea | S1-lineA |
| CASA | N | within_macro_eer | 100 | -3.443 | -11.37 | 2.024 | 30 | xlsr300m-linea | S1-lineA |
| CASA | E | pooled_eer | 100 | -0.2336 | -4.627 | 3.37 | 30 | xlsr300m-linea | S1-lineA |
| CASA | E | within_macro_eer | 100 | 0.08276 | -4.678 | 4.978 | 30 | xlsr300m-linea | S1-lineA |
| CASA | K | pooled_eer | 100 | -5.25 | -8.543 | -2.04 | 30 | xlsr300m-linea | S1-lineA |
| CASA | K | within_macro_eer | 100 | -5.575 | -12 | -1.235 | 30 | xlsr300m-linea | S1-lineA |
| CASA | BADGE | pooled_eer | 100 | -2.138 | -6.74 | 1.24 | 30 | xlsr300m-linea | S1-lineA |
| CASA | BADGE | within_macro_eer | 100 | -2.022 | -6.762 | 4.758 | 30 | xlsr300m-linea | S1-lineA |
| CASA | U | pooled_eer | 290 | -5.799 | -9.201 | -2.003 | 30 | xlsr300m-linea | S1-lineA |
| CASA | U | within_macro_eer | 290 | -5.631 | -8.358 | -1.723 | 30 | xlsr300m-linea | S1-lineA |
| CASA | P | pooled_eer | 290 | -4.211 | -8.054 | -1.289 | 30 | xlsr300m-linea | S1-lineA |
| CASA | P | within_macro_eer | 290 | -4.394 | -7.547 | -1.713 | 30 | xlsr300m-linea | S1-lineA |
| CASA | N | pooled_eer | 290 | -3.972 | -7.096 | -1.273 | 30 | xlsr300m-linea | S1-lineA |
| CASA | N | within_macro_eer | 290 | -4.328 | -7.347 | -2.288 | 30 | xlsr300m-linea | S1-lineA |
| CASA | E | pooled_eer | 290 | -0.02713 | -1.563 | 1.131 | 30 | xlsr300m-linea | S1-lineA |
| CASA | E | within_macro_eer | 290 | -0.04881 | -1.375 | 1.524 | 30 | xlsr300m-linea | S1-lineA |
| CASA | K | pooled_eer | 290 | -8.407 | -9.823 | -7.077 | 30 | xlsr300m-linea | S1-lineA |
| CASA | K | within_macro_eer | 290 | -7.129 | -8.749 | -5.246 | 30 | xlsr300m-linea | S1-lineA |
| CASA | BADGE | pooled_eer | 290 | -2.526 | -4.819 | -0.2537 | 30 | xlsr300m-linea | S1-lineA |
| CASA | BADGE | within_macro_eer | 290 | -3.73 | -9.775 | -1.153 | 30 | xlsr300m-linea | S1-lineA |


## W13 B10

NOT_RUN - `results/w13_b10.csv` missing

## W14 GraphSSL

NOT_RUN - `results/w14_graphssl.csv` missing

## W15 ITW

NOT_RUN - `results/w15_itw.csv` missing

## W16 factorial

2^3 factor matrix.

| pooled_eer | within_macro_eer | worst_platform | worst_platform_eer | auc | hter_pooled | n_eval | n_pos | eer_bilibili | eer_facebook | eer_rednote | eer_tiktok | eer_x | eer_youtube | protocol | region | backbone | spec | method | c1 | past | casa | k | draw | mask | n_labeled | eer_douyin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20.38 | 18.46 | youtube | 28.17 | 0.8869 | nan | 7336 | 1747 | 13.57 | 6.797 | 13.9 | 25.75 | 22.57 | 28.17 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 0 | clips | 100 | nan |
| 30.4 | 23.16 | youtube | 38.95 | 0.7798 | nan | 1661 | 451 | nan | nan | 17.13 | 13.38 | nan | 38.95 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 0 | strict | 100 | nan |
| 23.17 | 28.33 | douyin | 57.02 | 0.8592 | nan | 7336 | 1748 | 39.24 | 8.564 | 17.41 | 19.61 | 22.57 | 33.93 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 1 | clips | 100 | 57.02 |
| 29.39 | 23.11 | bilibili | 39.96 | 0.7913 | nan | 1888 | 558 | 39.96 | nan | 17.14 | 9.148 | nan | 26.2 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 1 | strict | 100 | nan |
| 17.61 | 19.17 | x | 26.77 | 0.9028 | nan | 7336 | 1749 | 18.76 | 9.584 | 12.15 | 22.67 | 26.77 | 25.08 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 2 | clips | 100 | nan |
| 22.18 | 26.45 | facebook | 46.43 | 0.8533 | nan | 1052 | 216 | nan | 46.43 | 14 | 22.53 | nan | 22.85 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 2 | strict | 100 | nan |
| 16.36 | 20.34 | douyin | 43.8 | 0.9105 | nan | 7336 | 1749 | 13.06 | 6.981 | 11.47 | 15.45 | 17.45 | 34.21 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 3 | clips | 100 | 43.8 |
| 26.57 | 21.01 | youtube | 52.46 | 0.8244 | nan | 1449 | 606 | nan | nan | 5.391 | 5.196 | nan | 52.46 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 3 | strict | 100 | nan |
| 14.63 | 16.61 | tiktok | 23.2 | 0.9113 | nan | 7336 | 1750 | 14.99 | 6.725 | 12.01 | 23.2 | 19.9 | 22.81 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 4 | clips | 100 | nan |
| 26.27 | 25.77 | facebook | 43.89 | 0.8207 | nan | 1797 | 236 | 27.49 | 43.89 | nan | 10.11 | 16.44 | 30.93 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 4 | strict | 100 | nan |
| 20.92 | 20.99 | youtube | 29.56 | 0.8764 | nan | 7336 | 1754 | 25.84 | 8.417 | 14.17 | 26.8 | 21.13 | 29.56 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 5 | clips | 100 | nan |
| 21.43 | 21.88 | rednote | 29.28 | 0.852 | nan | 1796 | 598 | 12.94 | nan | 29.28 | 16.23 | nan | 29.08 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 5 | strict | 100 | nan |
| 23.51 | 23.49 | youtube | 46.21 | 0.8525 | nan | 7336 | 1748 | 32.02 | 6.805 | 12.96 | 22.67 | 20.28 | 46.21 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 6 | clips | 100 | nan |
| 25.49 | 21.6 | youtube | 38.23 | 0.8541 | nan | 1588 | 369 | 30.53 | nan | 6.311 | 11.32 | nan | 38.23 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 6 | strict | 100 | nan |
| 14.6 | 16.05 | bilibili | 20.12 | 0.9189 | nan | 7336 | 1746 | 20.12 | 7.314 | 10.8 | 20.09 | 19.44 | 18.52 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 7 | clips | 100 | nan |
| 25.46 | 23.58 | facebook | 34.46 | 0.8188 | nan | 818 | 196 | nan | 34.46 | 26.45 | 7.113 | nan | 26.31 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 7 | strict | 100 | nan |
| 22.09 | 22.85 | bilibili | 33.03 | 0.8565 | nan | 7336 | 1752 | 33.03 | 9.497 | 14.04 | 28.89 | 21.52 | 30.13 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 8 | clips | 100 | nan |
| 35.93 | 34.5 | bilibili | 49.84 | 0.6883 | nan | 2109 | 245 | 49.84 | nan | 24.38 | 29.28 | nan | nan | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 8 | strict | 100 | nan |
| 22.36 | 24.53 | youtube | 45.78 | 0.8637 | nan | 7336 | 1749 | 26.31 | 9.32 | 14.44 | 28.34 | 20.3 | 45.78 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 9 | clips | 100 | 27.27 |
| 30.75 | 32.78 | youtube | 48.27 | 0.7793 | nan | 2004 | 520 | 34.41 | nan | 9.12 | 39.3 | nan | 48.27 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 9 | strict | 100 | nan |
| 23.86 | 23.92 | bilibili | 37.94 | 0.8328 | nan | 7336 | 1748 | 37.94 | 10.16 | 13.5 | 23.2 | 26.02 | 32.68 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 10 | clips | 100 | nan |
| 37.66 | 24.14 | youtube | 35.52 | 0.7003 | nan | 1566 | 428 | nan | 15.01 | 23.16 | 22.87 | nan | 35.52 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 10 | strict | 100 | nan |
| 25.59 | 26.87 | youtube | 40.25 | 0.8269 | nan | 7336 | 1747 | 36.57 | 16.29 | 21.05 | 26.83 | 20.22 | 40.25 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 11 | clips | 100 | nan |
| 37.26 | 30 | youtube | 49.2 | 0.6993 | nan | 2279 | 754 | 36 | nan | 19.34 | 15.45 | nan | 49.2 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 11 | strict | 100 | nan |
| 17.55 | 18.52 | x | 23.58 | 0.8889 | nan | 7336 | 1744 | 21.58 | 8.396 | 13.09 | 22.64 | 23.58 | 21.83 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 12 | clips | 100 | nan |
| 28.04 | 25.24 | facebook | 32.38 | 0.7792 | nan | 1776 | 467 | nan | 32.38 | 17.98 | 24.75 | nan | 25.85 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 12 | strict | 100 | nan |
| 26.21 | 25.99 | youtube | 40.02 | 0.8157 | nan | 7336 | 1755 | 38.93 | 8.652 | 14.17 | 24.25 | 29.9 | 40.02 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 13 | clips | 100 | nan |
| 31.9 | 31.6 | youtube | 45.79 | 0.7469 | nan | 1693 | 520 | nan | nan | 12.98 | 36.02 | nan | 45.79 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 13 | strict | 100 | nan |
| 13.18 | 14.51 | youtube | 23.21 | 0.9404 | nan | 7336 | 1752 | 12.78 | 8.822 | 12.42 | 17.52 | 12.29 | 23.21 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 14 | clips | 100 | nan |
| 19.65 | 19 | youtube | 31.7 | 0.8803 | nan | 1410 | 554 | 9.655 | nan | 20.51 | 14.16 | nan | 31.7 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 14 | strict | 100 | nan |
| 18.69 | 24.32 | douyin | 53.31 | 0.9027 | nan | 7336 | 1745 | 27.42 | 6.819 | 14.31 | 25.23 | 19.79 | 23.38 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 15 | clips | 100 | 53.31 |
| 17.65 | 22.14 | tiktok | 29.34 | 0.9073 | nan | 1354 | 589 | nan | nan | 15.63 | 29.34 | 23.33 | 20.24 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 15 | strict | 100 | nan |
| 19.02 | 20.51 | youtube | 30.37 | 0.8915 | nan | 7336 | 1751 | 21.49 | 9.062 | 14.04 | 26.83 | 21.24 | 30.37 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 16 | clips | 100 | nan |
| 23.38 | 24.73 | youtube | 31.56 | 0.842 | nan | 2396 | 714 | 15.48 | nan | 22.59 | 29.28 | nan | 31.56 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 16 | strict | 100 | nan |
| 16.22 | 15.99 | bilibili | 23.99 | 0.9188 | nan | 7336 | 1746 | 23.99 | 7.634 | 9.177 | 17.54 | 15.14 | 22.49 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 18 | clips | 100 | nan |
| 15.89 | 17.19 | youtube | 28.61 | 0.9028 | nan | 1411 | 308 | nan | nan | 5.052 | 17.9 | nan | 28.61 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 18 | strict | 100 | nan |
| 18.1 | 19.19 | x | 28.73 | 0.8938 | nan | 7336 | 1756 | 23.37 | 9.303 | 12.42 | 22.16 | 28.73 | 19.17 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 17 | clips | 100 | nan |
| 30.33 | 29.26 | bilibili | 43.99 | 0.7628 | nan | 1339 | 412 | 43.99 | nan | 26.84 | 22.05 | nan | 24.18 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 17 | strict | 100 | nan |
| 20.69 | 24.4 | douyin | 43.39 | 0.8731 | nan | 7336 | 1750 | 27.08 | 9.572 | 16.19 | 23.72 | 23.07 | 27.78 | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 19 | clips | 100 | 43.39 |
| 33.55 | 21.62 | rednote | 27.5 | 0.7086 | nan | 1019 | 170 | nan | nan | 27.5 | 15.73 | nan | nan | dev_v2 | dev | xlsr300m-linea | S1-lineA | C1=0|PAST=0|CASA=0 | 0 | 0 | 0 | 100 | 19 | strict | 100 | nan |
| _(920 more rows in the CSV)_ | | | | | | | | | | | | | | | | | | | | | | | | | | |


## W16 mechanism

De-confounding evidence.

| protocol | region | backbone | spec | method | metric | value | k | draw |
|---|---|---|---|---|---|---|---|---|
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6304 | 290 | 1 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 17.33 | 290 | 1 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6305 | 290 | 1 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 17.55 | 290 | 1 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6328 | 290 | 0 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 12.75 | 290 | 0 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6316 | 290 | 0 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 12.69 | 290 | 0 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6188 | 290 | 2 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 12.73 | 290 | 2 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6147 | 290 | 2 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 12.98 | 290 | 2 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6152 | 290 | 4 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 11.84 | 290 | 4 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6167 | 290 | 4 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 11.59 | 290 | 4 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6341 | 290 | 3 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 12.74 | 290 | 3 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6347 | 290 | 3 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 13 | 290 | 3 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.603 | 290 | 5 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 14.09 | 290 | 5 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6032 | 290 | 5 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 14.4 | 290 | 5 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.625 | 290 | 6 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 11.71 | 290 | 6 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6199 | 290 | 6 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 11.76 | 290 | 6 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.59 | 290 | 7 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 11.37 | 290 | 7 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.5858 | 290 | 7 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 11.53 | 290 | 7 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6129 | 290 | 9 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 15.39 | 290 | 9 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6074 | 290 | 9 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 15.79 | 290 | 9 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | platform_auc | 0.6153 | 290 | 10 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | B1 | within_macro_eer | 13.1 | 290 | 10 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | platform_auc | 0.6144 | 290 | 10 |
| dev_v2 | dev | xlsr300m-linea | S1-lineA | C1 | within_macro_eer | 13.33 | 290 | 10 |
| _(80 more rows in the CSV)_ | | | | | | | | |


## W17 NDA/extra backbones

NOT_RUN - `results/w17_nda.csv` missing

## W18 C4 ensemble

E1 - E3.

| method | other | metric | arm | mean | lo | hi | n | backbone | spec | note |
|---|---|---|---|---|---|---|---|---|---|---|
| C1+TTA | C1 | pooled_eer | E2 | 0.7665 | -0.01982 | 1.66 | 50 | xls-r-2b | S3 | nan |
| C1+TTA | C1 | within_macro_eer | E2 | 0.6364 | -0.02416 | 1.86 | 50 | xls-r-2b | S3 | nan |
| PAST+TTA | PAST | pooled_eer | E2 | 0.6195 | -0.04366 | 1.324 | 50 | xls-r-2b | S3 | nan |
| PAST+TTA | PAST | within_macro_eer | E2 | 0.6163 | -0.3529 | 1.667 | 50 | xls-r-2b | S3 | nan |
| E1 | top-2 backbones | pooled_eer | E1 | nan | nan | nan | 0 | xls-r-2b | S3 | NOT_RUN(needs_2nd_backbone) |
| E3 | E1+E2 | pooled_eer | E3 | nan | nan | nan | 0 | xls-r-2b | S3 | NOT_RUN(needs_2nd_backbone) |


## W19 Fam-B robustness

Unseen transcoding chain.

| method | clean | famb | dEER | backbone | spec |
|---|---|---|---|---|---|
| B1 | 7.443 | 8.127 | 0.6834 | xls-r-2b | S3 |
| B2 | 11.94 | 8.127 | -3.813 | xls-r-2b | S3 |
| C1 | 7.748 | 8.361 | 0.6126 | xls-r-2b | S3 |
| PAST | 7.434 | 7.932 | 0.4983 | xls-r-2b | S3 |
| PAST+TTA | 8.053 | 8.264 | 0.2105 | xls-r-2b | S3 |


## W20 sensitivity

Grid and setting sensitivity.

| knob | value | method | pooled_eer_mean | pooled_eer_std | within_macro_eer_mean | within_macro_eer_std | worst_platform_eer_mean | worst_platform_eer_std | auc_mean | auc_std | backbone | spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell | language | B1 | 7.477 | 0.8436 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9737 | 0.003607 | xls-r-2b | S3 |
| cell | language | B2 | 12.48 | 2.766 | 11.65 | 1.663 | 22.86 | 7.286 | 0.9345 | 0.02239 | xls-r-2b | S3 |
| cell | language | C1 | 7.792 | 0.9251 | 9.095 | 1.225 | 16.57 | 4.406 | 0.9709 | 0.003901 | xls-r-2b | S3 |
| cell | language | PAST | 7.865 | 0.9013 | 9.192 | 1.211 | 16.74 | 4.307 | 0.9699 | 0.004073 | xls-r-2b | S3 |
| cell | platform | B1 | 7.477 | 0.8436 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9737 | 0.003607 | xls-r-2b | S3 |
| cell | platform | B2 | 8.003 | 1.559 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9669 | 0.009045 | xls-r-2b | S3 |
| cell | platform | C1 | 7.793 | 0.9257 | 9.094 | 1.226 | 16.57 | 4.406 | 0.9709 | 0.003902 | xls-r-2b | S3 |
| cell | platform | PAST | 7.551 | 0.8796 | 9.138 | 1.214 | 16.64 | 4.502 | 0.9726 | 0.003638 | xls-r-2b | S3 |
| cell | platform_x_language | B1 | 7.477 | 0.8436 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9737 | 0.003607 | xls-r-2b | S3 |
| cell | platform_x_language | B2 | 12.17 | 2.992 | 10.24 | 1.182 | 17.37 | 5.363 | 0.945 | 0.01686 | xls-r-2b | S3 |
| cell | platform_x_language | C1 | 7.79 | 0.9247 | 9.096 | 1.225 | 16.57 | 4.406 | 0.9709 | 0.003901 | xls-r-2b | S3 |
| cell | platform_x_language | PAST | 7.527 | 0.8833 | 9.128 | 1.261 | 16.72 | 4.834 | 0.9726 | 0.003816 | xls-r-2b | S3 |
| label_draw | global | B1 | 6.749 | 0.9293 | 9.067 | 1.64 | 18.13 | 6.878 | 0.9789 | 0.003675 | xls-r-2b | S3 |
| label_draw | global | B2 | 11.31 | 1.913 | 10.14 | 1.749 | 18.72 | 7.097 | 0.9496 | 0.01307 | xls-r-2b | S3 |
| label_draw | global | C1 | 6.857 | 0.9709 | 9.197 | 1.636 | 18.31 | 6.764 | 0.9773 | 0.00423 | xls-r-2b | S3 |
| label_draw | global | PAST | 6.651 | 1.032 | 9.179 | 1.678 | 18.54 | 7.494 | 0.9779 | 0.004406 | xls-r-2b | S3 |
| label_draw | per_cell | B1 | 7.477 | 0.8436 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9737 | 0.003607 | xls-r-2b | S3 |
| label_draw | per_cell | B2 | 12.17 | 2.992 | 10.24 | 1.182 | 17.37 | 5.363 | 0.945 | 0.01686 | xls-r-2b | S3 |
| label_draw | per_cell | C1 | 7.79 | 0.9247 | 9.096 | 1.225 | 16.57 | 4.406 | 0.9709 | 0.003901 | xls-r-2b | S3 |
| label_draw | per_cell | PAST | 7.527 | 0.8833 | 9.128 | 1.261 | 16.72 | 4.834 | 0.9726 | 0.003816 | xls-r-2b | S3 |
| label_draw | speaker_clustered | B1 | 11.55 | 1.953 | 12.7 | 2.155 | 22.78 | 6.057 | 0.9537 | 0.01258 | xls-r-2b | S3 |
| label_draw | speaker_clustered | B2 | 19.62 | 4.941 | 14.13 | 2.314 | 25.54 | 6.718 | 0.8805 | 0.03906 | xls-r-2b | S3 |
| label_draw | speaker_clustered | C1 | 11.8 | 2.226 | 13 | 2.342 | 23.22 | 6.598 | 0.9511 | 0.01516 | xls-r-2b | S3 |
| label_draw | speaker_clustered | PAST | 10.88 | 1.768 | 12.46 | 2.171 | 22.46 | 7.076 | 0.9569 | 0.0115 | xls-r-2b | S3 |
| label_noise | 0.0 | B1 | 7.477 | 0.8436 | 8.684 | 1.188 | 16.23 | 5.122 | 0.9737 | 0.003607 | xls-r-2b | S3 |
| label_noise | 0.0 | B2 | 12.17 | 2.992 | 10.24 | 1.182 | 17.37 | 5.363 | 0.945 | 0.01686 | xls-r-2b | S3 |
| label_noise | 0.0 | C1 | 7.79 | 0.9247 | 9.096 | 1.225 | 16.57 | 4.406 | 0.9709 | 0.003901 | xls-r-2b | S3 |
| label_noise | 0.0 | PAST | 7.527 | 0.8833 | 9.128 | 1.261 | 16.72 | 4.834 | 0.9726 | 0.003816 | xls-r-2b | S3 |
| label_noise | 0.05 | B1 | 8.838 | 1.253 | 9.703 | 1.399 | 17.52 | 5.783 | 0.9678 | 0.006282 | xls-r-2b | S3 |
| label_noise | 0.05 | B2 | 12.06 | 3.077 | 11 | 1.623 | 18.26 | 5.582 | 0.9462 | 0.01776 | xls-r-2b | S3 |
| label_noise | 0.05 | C1 | 12.33 | 3.097 | 13.55 | 3.111 | 22.4 | 6.78 | 0.9411 | 0.02143 | xls-r-2b | S3 |
| label_noise | 0.05 | PAST | 11.61 | 3.066 | 13.27 | 2.958 | 22.08 | 6.865 | 0.9457 | 0.0205 | xls-r-2b | S3 |
| label_noise | 0.1 | B1 | 9.583 | 1.262 | 10.63 | 1.584 | 19.4 | 5.487 | 0.9636 | 0.0083 | xls-r-2b | S3 |
| label_noise | 0.1 | B2 | 12.15 | 2.613 | 11.82 | 1.912 | 19.62 | 5.352 | 0.9454 | 0.0173 | xls-r-2b | S3 |
| label_noise | 0.1 | C1 | 16.8 | 3.176 | 17.95 | 3.116 | 28.41 | 7.241 | 0.9055 | 0.02748 | xls-r-2b | S3 |
| label_noise | 0.1 | PAST | 15.63 | 3.116 | 17.24 | 3.068 | 28.06 | 7.528 | 0.9149 | 0.02629 | xls-r-2b | S3 |
| pca | 128 | B1 | 7.689 | 0.9053 | 9.03 | 1.334 | 16.82 | 5.785 | 0.9724 | 0.003803 | xls-r-2b | S3 |
| pca | 128 | B2 | 12.41 | 3.113 | 10.51 | 1.26 | 17.83 | 5.92 | 0.9435 | 0.01723 | xls-r-2b | S3 |
| pca | 128 | C1 | 8.696 | 1.194 | 10.15 | 1.584 | 18.07 | 5.163 | 0.9655 | 0.005472 | xls-r-2b | S3 |
| pca | 128 | PAST | 8.362 | 1.152 | 10.11 | 1.55 | 18.11 | 5.225 | 0.9675 | 0.00509 | xls-r-2b | S3 |
| _(12 more rows in the CSV)_ | | | | | | | | | | | | |


## W21 ADD-GP

NOT_RUN - `results/w21_addgp.csv` missing

## 5. Reproduction and commit

- commit: `d8f7f8e`
- environment: `scripts/w1_env.py`; queue: `queue.yaml`; criteria: `claims.yaml`
