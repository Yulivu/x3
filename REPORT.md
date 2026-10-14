# X3 report

```
DECISION: G0=NOT_RUN G1=NOT_RUN G2=REPRODUCED G3=NOT_RUN G4=C1_FAIL G5=PL_OK G6=PAST_FAIL G7=CASA_PASS G8=NOT_RUN G9=NOT_RUN G10=NOT_RUN G11=NOT_RUN
```

## 1. Summary

- S-F (few-shot SOTA): NOT_RUN
- S-Z (zero-label SOTA): NOT_RUN
- N1 (MPP / C1): C1_FAIL
- N2 (PAST): PAST_FAIL
- N3 (CASA): CASA_PASS
- N4 (de-confounding): NOT_RUN
- S-K (budget curve): see results/w7_curve.csv
- S-R (ITW replication): NOT_RUN

Novelty rule: 1/4 of N1-N4 passed -> one component passed: report it as an implementation detail.

- {'item': 'TASKBOOK 1.2 headline attribution', 'what': "TASKBOOK says Line A's headline 13.64/14.56/20.79 comes from 'frozen XLS-R features + a small probe trained on 290 labels, plus cell-prior correction'. Measured on the box, that route gives 24.65 pooled (k=10, BB+EB), and even an ORACLE cell prior only reaches 21.52 pooled on the p0cal scores. The 13.64 number is COMPARISON.md's line A: 290 labels train a head directly with NO prior correction (clips 13.05 / strict 13.64, AUC 0.94).", 'impact': "S-F must be judged against the ADAPTATION route (BASE_F's B11 family), not only against the prior-correction family; otherwise the SOTA bar is ~9 EER points too low. The prior-correction family stays in BASE_F under its own name.", 'source': 'W4 on the box, 2026-10-14; wx2/docs/COMPARISON.md 0.0 line 42'}

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

NOT_RUN - `results/w7_curve.csv` missing

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

NOT_RUN - `results/w16_factorial.csv` missing

## W16 mechanism

NOT_RUN - `results/w16_mechanism.csv` missing

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

- commit: `f0bec98`
- environment: `scripts/w1_env.py`; queue: `queue.yaml`; criteria: `claims.yaml`
