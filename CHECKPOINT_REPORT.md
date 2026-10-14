# X3 checkpoint report

```
DECISION: G0=pass G1=HEADROOM_OK G2=REPRODUCED G3=xls-r-2b|S3 G4=C1_FAIL G5=PL_OK G6=PAST_FAIL G7=CASA_PASS G8=Z_FAIL G9=NOT_RUN G10=DECONF_NOT_SHOWN G11=SOTA_F_DEV_NO G12=MALP_INFO G13=SEL_FAIL G14=SPEAKER_COVERAGE_FAIL G15=LABEL_EFFICIENCY_PASS G16=SOTA_F2_DEV_NO G17=ESN_FAIL G18=PROJ_FAIL
```
FLAGS: SPEAKER_GAP=False LP_GAIN_IS_SPEAKER_LOCAL=False B11_BROKEN=True IDENTITY_OVERLAP_FLAG=False

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

## Work packages

_(empty)_

## Headline tables

### W2 zero-shot

`results/zeroshot_mlitw.csv`

| pooled_eer | within_macro_eer | worst_platform | worst_platform_eer | auc | hter_pooled | n_eval | n_pos | eer_bilibili | eer_douyin | eer_facebook | eer_rednote | eer_tiktok | eer_x | eer_youtube | backbone | kind | fidelity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20.21 | 21.52 | tiktok | 23.89 | 0.8627 | 35.6 | 24529 | 6361 | 21.91 | 19.23 | 22.5 | 23.62 | 23.89 | 18.22 | 21.27 | xls-r-2b | antideepfake | pass |
| 22.62 | 22.04 | rednote | 27.62 | 0.8501 | 35.73 | 24529 | 6361 | 24.59 | 23.2 | 20.91 | 27.62 | 20.33 | 19.33 | 18.28 | xls-r-1b | antideepfake | pass |
| 26.32 | 23.59 | douyin | 29.76 | 0.82 | 39.62 | 24529 | 6361 | 23.08 | 29.76 | 20.47 | 29.57 | 22.99 | 20.28 | 19 | mms-1b | antideepfake | pass |
| 32.26 | 50 | bilibili | 50 | 0.7531 | 34.19 | 24529 | 6361 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | metadata-lookup | metadata | nan |

### W4 reconciliation

`results/reconciliation.csv`

| scope | metric | headline | rerun | diff | mask | n_draws | reference |
|---|---|---|---|---|---|---|---|
| 3draw | pooled_eer | 24.65 | 25.85 | 1.198 | clips | 3 | fsc_var_results.csv (bb, k=10) |
| 3draw | within_macro_eer | 34.36 | 35.07 | 0.7102 | clips | 3 | fsc_var_results.csv (bb, k=10) |
| 100draw | pooled_eer | 24.65 | 25.1 | 0.4479 | clips | 100 | fsc_var_results.csv (bb, k=10) |
| 100draw | within_macro_eer | 34.36 | 34.24 | -0.1162 | clips | 100 | fsc_var_results.csv (bb, k=10) |

### W4 mechanism

`results/w4_mechanism.csv`

| method | mask | pooled_eer_mean | pooled_eer_std | within_macro_eer_mean | within_macro_eer_std | worst_platform_eer_mean | worst_platform_eer_std | auc_mean | auc_std |
|---|---|---|---|---|---|---|---|---|---|
| M0 | clips | 32.43 | 1.128 | 36.54 | 1.533 | 44.65 | 1.328 | 0.7469 | 0.01585 |
| M0 | strict | 30.02 | 2.567 | 35.81 | 3.776 | 50.97 | 8.038 | 0.7706 | 0.03263 |
| M1 | clips | 30.36 | 2.27 | 44.4 | 1.442 | 52.06 | 3.233 | 0.7627 | 0.01505 |
| M1 | strict | 25.77 | 3.856 | 50.37 | 3.747 | 64.61 | 15.27 | 0.7919 | 0.03611 |
| M2 | clips | 25.1 | 1.494 | 34.24 | 1.771 | 44.65 | 1.328 | 0.8281 | 0.01316 |
| M2 | strict | 22.63 | 2.395 | 37.25 | 4.193 | 56.02 | 13.25 | 0.8474 | 0.02415 |

### W5 backbone rank

`results/backbone_rank.csv`

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
| _(12 more rows in the CSV)_ | | | | | | | | | |

### W6 summary

`results/w6_summary.csv`

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
| _(60 more rows in the CSV)_ | | | | | | |

### W6 paired

`results/w6_paired.csv`

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

### W7 budget curve

`results/w7_curve.csv`

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
| _(18 more rows in the CSV)_ | | | | | | | | | | | |

### W8 Track Z

`results/w8_zlabel.csv`

| pooled_eer | within_macro_eer | worst_platform | worst_platform_eer | auc | hter_pooled | n_eval | n_pos | eer_bilibili | eer_douyin | eer_facebook | eer_rednote | eer_tiktok | eer_x | eer_youtube | protocol | region | backbone | spec | method | k | mask | n_labeled | mu0 | mu1 | sig0 | sig1 | pi_global | loglik | seed_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20.86 | 21.54 | tiktok | 30.03 | 0.8573 | nan | 15002 | 3990 | 17.2 | 18.34 | 25.2 | 21.42 | 30.03 | 17.84 | 20.8 | track_z_v3_dev | dev | xls-r-2b | S3 | Z0 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | nan |
| 25.62 | 21.54 | tiktok | 30.03 | 0.8127 | nan | 15002 | 3990 | 17.2 | 18.34 | 25.2 | 21.42 | 30.03 | 17.84 | 20.8 | track_z_v3_dev | dev | xls-r-2b | S3 | Z1 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | nan |
| 22.18 | 23.33 | tiktok | 33.25 | 0.8373 | nan | 15002 | 3990 | 17.2 | 18.34 | 27.17 | 26.24 | 33.25 | 18.04 | 23.08 | track_z_v3_dev | dev | xls-r-2b | S3 | Z2 | 0 | clips | 0 | -105.3 | 0.3661 | 12.2 | 74.7 | 0.2642 | -4.777 | nan |
| 16.16 | 18.41 | tiktok | 36.73 | 0.8941 | nan | 15002 | 3990 | 17.14 | 10.92 | 13.57 | 20.07 | 36.73 | 14.56 | 15.9 | track_z_v3_dev | dev | xls-r-2b | S3 | Z5 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | Z2 |
| 23.74 | 21.04 | tiktok | 31.02 | 0.8508 | nan | 15002 | 3990 | 20.8 | 14.82 | 29.57 | 15.66 | 31.02 | 16.17 | 19.19 | track_z_v3_dev | dev | xls-r-2b | S3 | Z5-z1 | 0 | clips | 0 | nan | nan | nan | nan | nan | nan | Z1 |

commit: `d1e2ca5`
