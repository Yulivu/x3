# X3 report

```
DECISION: G0=NOT_RUN G1=NOT_RUN G2=REPRODUCED G3=NOT_RUN G4=NOT_RUN G5=PL_OK G6=NOT_RUN G7=NOT_RUN G8=NOT_RUN G9=NOT_RUN G10=NOT_RUN G11=NOT_RUN
```

## 1. Summary

- S-F (few-shot SOTA): NOT_RUN
- S-Z (zero-label SOTA): NOT_RUN
- N1 (MPP / C1): NOT_RUN
- N2 (PAST): NOT_RUN
- N3 (CASA): NOT_RUN
- N4 (de-confounding): NOT_RUN
- S-K (budget curve): see results/w7_curve.csv
- S-R (ITW replication): NOT_RUN

Novelty rule: 0/4 of N1-N4 passed -> method gain not shown: report the protocol and baseline results only.

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

| backbone | spec | method | mask | pooled_eer_mean | pooled_eer_std | within_macro_eer_mean | within_macro_eer_std | worst_platform_eer_mean | worst_platform_eer_std | auc_mean | auc_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xlsr300m-linea | S1-lineA | B1 | clips | 12.99 | 1.759 | 14.56 | 2.385 | 26.84 | 8.662 | 0.9373 | 0.0132 |
| xlsr300m-linea | S1-lineA | B1 | strict | 21.5 | 5.66 | 20.94 | 7.628 | 22.46 | 8.478 | 0.8596 | 0.05708 |
| xlsr300m-linea | S1-lineA | B11 | clips | 26.01 | 1.772 | 26.97 | 1.889 | 39.15 | 5.215 | 0.8157 | 0.01874 |
| xlsr300m-linea | S1-lineA | B11 | strict | 38.22 | 5.17 | 40.53 | 7.887 | 41.72 | 8.341 | 0.6634 | 0.06597 |
| xlsr300m-linea | S1-lineA | B2 | clips | 18.72 | 3.825 | 15.87 | 2.426 | 26.5 | 8.869 | 0.8974 | 0.02636 |
| xlsr300m-linea | S1-lineA | B2 | strict | 26.11 | 7.338 | 21.65 | 7.877 | 23.26 | 9.308 | 0.7984 | 0.08098 |
| xlsr300m-linea | S1-lineA | B3 | clips | 13.92 | 2.228 | 14.56 | 2.385 | 26.84 | 8.662 | 0.9318 | 0.01586 |
| xlsr300m-linea | S1-lineA | B3 | strict | 22.1 | 6.068 | 20.94 | 7.628 | 22.46 | 8.478 | 0.8483 | 0.06435 |
| xlsr300m-linea | S1-lineA | B5 | clips | 18.96 | 1.955 | 20.29 | 2.196 | 33.39 | 7.848 | 0.8872 | 0.01495 |
| xlsr300m-linea | S1-lineA | B5 | strict | 28.99 | 6.003 | 28.34 | 9.131 | 29.44 | 9.609 | 0.774 | 0.06865 |
| xlsr300m-linea | S1-lineA | B6 | clips | 33.82 | 1.265 | 27.08 | 2.251 | 39.23 | 12.99 | 0.7291 | 0.01508 |
| xlsr300m-linea | S1-lineA | B6 | strict | 36.08 | 7.506 | 32.29 | 8.599 | 34.57 | 9.371 | 0.6646 | 0.09555 |
| xlsr300m-linea | S1-lineA | B7 | clips | 12.35 | 1.997 | 13.89 | 2.397 | 25.03 | 8.411 | 0.9406 | 0.0149 |
| xlsr300m-linea | S1-lineA | B7 | strict | 21.28 | 6.054 | 21.07 | 8.05 | 22.66 | 9.039 | 0.8604 | 0.06132 |
| xlsr300m-linea | S1-lineA | B9 | clips | 13.78 | 2.254 | 15.06 | 2.313 | 25.88 | 6.277 | 0.9202 | 0.01209 |
| xlsr300m-linea | S1-lineA | B9 | strict | 33.73 | 9.37 | 33.58 | 13.16 | 36 | 16.16 | 0.6824 | 0.09138 |
| xlsr300m-linea | S1-lineA | C1 | clips | 13.41 | 1.896 | 14.8 | 2.378 | 25.82 | 8.269 | 0.9339 | 0.01454 |
| xlsr300m-linea | S1-lineA | C1 | strict | 22.61 | 5.861 | 21.89 | 7.882 | 23.57 | 9.031 | 0.8451 | 0.06097 |


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

NOT_RUN - `results/w9_audit.csv` missing

## W11 PAST

NOT_RUN - `results/w11_past.csv` missing

## W12 CASA

NOT_RUN - `results/w12_casa.csv` missing

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

- commit: `73c4b7f`
- environment: `scripts/w1_env.py`; queue: `queue.yaml`; criteria: `claims.yaml`
