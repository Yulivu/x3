# X3 report

```
DECISION: G0=NOT_RUN G1=NOT_RUN G2=REPRODUCED G3=NOT_RUN G4=C1_FAIL G5=PL_OK G6=PAST_FAIL G7=CASA_PASS G8=NOT_RUN G9=NOT_RUN G10=DECONF_NOT_SHOWN G11=NOT_RUN
```

## 1. Summary

- S-F (few-shot SOTA): NOT_RUN
- S-Z (zero-label SOTA): NOT_RUN
- N1 (MPP / C1): C1_FAIL
- N2 (PAST): PAST_FAIL
- N3 (CASA): CASA_PASS
- N4 (de-confounding): DECONF_NOT_SHOWN
- S-K (budget curve): see results/w7_curve.csv
- S-R (ITW replication): NOT_RUN

Novelty rule: 1/4 of N1-N4 passed -> one component passed: report it as an implementation detail.

- {'item': 'TASKBOOK 1.2 headline attribution', 'what': "TASKBOOK says Line A's headline 13.64/14.56/20.79 comes from 'frozen XLS-R features + a small probe trained on 290 labels, plus cell-prior correction'. Measured on the box, that route gives 24.65 pooled (k=10, BB+EB), and even an ORACLE cell prior only reaches 21.52 pooled on the p0cal scores. The 13.64 number is COMPARISON.md's line A: 290 labels train a head directly with NO prior correction (clips 13.05 / strict 13.64, AUC 0.94).", 'impact': "S-F must be judged against the ADAPTATION route (BASE_F's B11 family), not only against the prior-correction family; otherwise the SOTA bar is ~9 EER points too low. The prior-correction family stays in BASE_F under its own name.", 'source': 'W4 on the box, 2026-10-14; wx2/docs/COMPARISON.md 0.0 line 42'}
- {'item': 'B10_REDUCED is not a usable baseline', 'what': "B10_REDUCED scored mean AUC 0.412 (< 0.5) over 50 draws: the head never trained. With a random initialisation the declared budget (5 epochs, lr <= 3e-5, AdamW) leaves the weights essentially at their random values, so the score is a random projection dominated by PC1 and comes out anti-correlated with the label. The full B10 (W13 on the GPU lane) is supposed to start from the backbone's own pretrained head, which this variant cannot do.", 'impact': "Any 'we beat B10' statement made with these rows is vacuous. Either run the full B10 on the encoder, or report B10 as NOT_MEANINGFUL and keep it out of the S-F comparison.", 'source': 'W13 on the box, 2026-10-14'}

## W2 zero-shot

NOT_RUN - `results/zeroshot_mlitw.csv` missing

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

NOT_RUN - `results/backbone_rank.csv` missing

## W6 summary

k=290, every baseline plus C1.

| backbone | spec | method | mask | metric | mean | std |
|---|---|---|---|---|---|---|
| xlsr300m-linea | S1-lineA | B1 | clips | pooled_eer | 12.99 | 1.759 |
| xlsr300m-linea | S1-lineA | B1 | clips | within_macro_eer | 14.56 | 2.385 |
| xlsr300m-linea | S1-lineA | B1 | clips | worst_platform_eer | 26.84 | 8.662 |
| xlsr300m-linea | S1-lineA | B1 | clips | auc | 0.9373 | 0.0132 |
| xlsr300m-linea | S1-lineA | B1 | strict | pooled_eer | 21.5 | 5.66 |
| xlsr300m-linea | S1-lineA | B1 | strict | within_macro_eer | 20.94 | 7.628 |
| xlsr300m-linea | S1-lineA | B1 | strict | worst_platform_eer | 22.46 | 8.478 |
| xlsr300m-linea | S1-lineA | B1 | strict | auc | 0.8596 | 0.05708 |
| xlsr300m-linea | S1-lineA | B11 | clips | pooled_eer | 26.01 | 1.772 |
| xlsr300m-linea | S1-lineA | B11 | clips | within_macro_eer | 26.97 | 1.889 |
| xlsr300m-linea | S1-lineA | B11 | clips | worst_platform_eer | 39.15 | 5.215 |
| xlsr300m-linea | S1-lineA | B11 | clips | auc | 0.8157 | 0.01874 |
| xlsr300m-linea | S1-lineA | B11 | strict | pooled_eer | 38.22 | 5.17 |
| xlsr300m-linea | S1-lineA | B11 | strict | within_macro_eer | 40.53 | 7.887 |
| xlsr300m-linea | S1-lineA | B11 | strict | worst_platform_eer | 41.72 | 8.341 |
| xlsr300m-linea | S1-lineA | B11 | strict | auc | 0.6634 | 0.06597 |
| xlsr300m-linea | S1-lineA | B2 | clips | pooled_eer | 18.72 | 3.825 |
| xlsr300m-linea | S1-lineA | B2 | clips | within_macro_eer | 15.87 | 2.426 |
| xlsr300m-linea | S1-lineA | B2 | clips | worst_platform_eer | 26.5 | 8.869 |
| xlsr300m-linea | S1-lineA | B2 | clips | auc | 0.8974 | 0.02636 |
| xlsr300m-linea | S1-lineA | B2 | strict | pooled_eer | 26.11 | 7.338 |
| xlsr300m-linea | S1-lineA | B2 | strict | within_macro_eer | 21.65 | 7.877 |
| xlsr300m-linea | S1-lineA | B2 | strict | worst_platform_eer | 23.26 | 9.308 |
| xlsr300m-linea | S1-lineA | B2 | strict | auc | 0.7984 | 0.08098 |
| xlsr300m-linea | S1-lineA | B3 | clips | pooled_eer | 13.92 | 2.228 |
| xlsr300m-linea | S1-lineA | B3 | clips | within_macro_eer | 14.56 | 2.385 |
| xlsr300m-linea | S1-lineA | B3 | clips | worst_platform_eer | 26.84 | 8.662 |
| xlsr300m-linea | S1-lineA | B3 | clips | auc | 0.9318 | 0.01586 |
| xlsr300m-linea | S1-lineA | B3 | strict | pooled_eer | 22.1 | 6.068 |
| xlsr300m-linea | S1-lineA | B3 | strict | within_macro_eer | 20.94 | 7.628 |
| xlsr300m-linea | S1-lineA | B3 | strict | worst_platform_eer | 22.46 | 8.478 |
| xlsr300m-linea | S1-lineA | B3 | strict | auc | 0.8483 | 0.06435 |
| xlsr300m-linea | S1-lineA | B5 | clips | pooled_eer | 18.96 | 1.955 |
| xlsr300m-linea | S1-lineA | B5 | clips | within_macro_eer | 20.29 | 2.196 |
| xlsr300m-linea | S1-lineA | B5 | clips | worst_platform_eer | 33.39 | 7.848 |
| xlsr300m-linea | S1-lineA | B5 | clips | auc | 0.8872 | 0.01495 |
| xlsr300m-linea | S1-lineA | B5 | strict | pooled_eer | 28.99 | 6.003 |
| xlsr300m-linea | S1-lineA | B5 | strict | within_macro_eer | 28.34 | 9.131 |
| xlsr300m-linea | S1-lineA | B5 | strict | worst_platform_eer | 29.44 | 9.609 |
| xlsr300m-linea | S1-lineA | B5 | strict | auc | 0.774 | 0.06865 |
| _(32 more rows in the CSV)_ | | | | | | |


## W6 paired

C1 vs B1/B2/B3/B11, draw-paired.

| method | other | metric | mean | lo | hi | n | backbone | spec |
|---|---|---|---|---|---|---|---|---|
| C1 | B1 | pooled_eer | 0.7668 | -2.103 | 4.347 | 200 | xlsr300m-linea | S1-lineA |
| C1 | B1 | within_macro_eer | 0.5735 | -2.461 | 4.517 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B1 | worst_platform_eer | -0.01373 | -7.435 | 5.156 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B2 | pooled_eer | -4.406 | -14.17 | 6.766 | 200 | xlsr300m-linea | S1-lineA |
| C1 | B2 | within_macro_eer | -0.4551 | -4.484 | 4.497 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B2 | worst_platform_eer | -0.2147 | -8.843 | 4.992 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B3 | pooled_eer | -0.004444 | -5.056 | 6.771 | 200 | xlsr300m-linea | S1-lineA |
| C1 | B3 | within_macro_eer | 0.5735 | -2.461 | 4.517 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B3 | worst_platform_eer | -0.01373 | -7.435 | 5.156 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B11 | pooled_eer | -14.11 | -29.11 | -3.606 | 200 | xlsr300m-linea | S1-lineA |
| C1 | B11 | within_macro_eer | -15.22 | -36.75 | -3.829 | 189 | xlsr300m-linea | S1-lineA |
| C1 | B11 | worst_platform_eer | -15.6 | -36.66 | 2.582 | 189 | xlsr300m-linea | S1-lineA |


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

NOT_RUN - `results/w8_zlabel.csv` missing

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

| method | other | metric | mean | lo | hi | n | backbone | spec |
|---|---|---|---|---|---|---|---|---|
| PAST | C1 | pooled_eer | -0.6834 | -2.19 | 0.4498 | 50 | xlsr300m-linea | S1-lineA |
| PAST | C1 | within_macro_eer | -0.6052 | -1.725 | 0.5326 | 50 | xlsr300m-linea | S1-lineA |
| PAST | B7 | pooled_eer | 0.3469 | -1.572 | 1.675 | 50 | xlsr300m-linea | S1-lineA |
| PAST | B7 | within_macro_eer | 0.3184 | -0.898 | 1.608 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-noEM | pooled_eer | -0.3267 | -3.1 | 1.675 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-noEM | within_macro_eer | -0.1299 | -1.112 | 1.222 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-noQuota | pooled_eer | -0.5209 | -1.986 | 0.682 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-noQuota | within_macro_eer | -0.5729 | -1.604 | 0.5823 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-1round | pooled_eer | -0.5613 | -1.839 | 0.4276 | 50 | xlsr300m-linea | S1-lineA |
| PAST | PAST-1round | within_macro_eer | -0.542 | -1.381 | 0.5233 | 50 | xlsr300m-linea | S1-lineA |


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

NOT_RUN - `results/w18_c4.csv` missing

## W19 Fam-B robustness

NOT_RUN - `results/w19_famb.csv` missing

## W20 sensitivity

NOT_RUN - `results/w20_sensitivity.csv` missing

## W21 ADD-GP

NOT_RUN - `results/w21_addgp.csv` missing

## 5. Reproduction and commit

- commit: `25d6768`
- environment: `scripts/w1_env.py`; queue: `queue.yaml`; criteria: `claims.yaml`
