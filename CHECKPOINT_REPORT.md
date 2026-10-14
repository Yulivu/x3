# X3 checkpoint report

```
DECISION: G0=NOT_RUN G1=NOT_RUN G2=REPRODUCED G3=NOT_RUN G4=C1_FAIL G5=PL_OK G6=PAST_FAIL G7=CASA_PASS G8=NOT_RUN G9=NOT_RUN G10=DECONF_NOT_SHOWN G11=NOT_RUN
```

## Work packages

_(empty)_

## Headline tables

### W2 zero-shot

NOT_RUN - `results/zeroshot_mlitw.csv` missing

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

NOT_RUN - `results/backbone_rank.csv` missing

### W6 summary

`results/w6_summary.csv`

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
| _(60 more rows in the CSV)_ | | | | | | |

### W6 paired

`results/w6_paired.csv`

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

NOT_RUN - `results/w8_zlabel.csv` missing

commit: `25d6768`
