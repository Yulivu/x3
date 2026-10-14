# X3 DECISION POINT (NEW_TASKS 02)

Protocol: `splits/speaker_split_v3.json` (frozen), criteria `claims_v2.yaml` v2.1 (sha256 in `claims_v2.sha256`).  All numbers are dev-region numbers; **no test label has ever been read** (`results/split_v3_audit.csv`).

## 1. Best method vs B1 (k=290 unless the label says m=...)

| mask | best method | mean EER | B1 mean | delta | (a) interval | (b) interval | old rule | >= 2.0 pts and (b) clear of 0 |
|---|---|---|---|---|---|---|---|---|
| strict_draw | OC+B1@m=10 | 5.70 | 14.30 | -8.60 | [-9.10, -8.14] | [-10.23, -7.34] | [-13.96, -4.19] | YES |
| unseen | OC+B1@m=10 | 6.19 | 10.51 | -4.32 | [-5.40, -3.32] | [-4.97, -3.22] | [-20.00, 0.07] | YES |
| seen | OC+B1@m=10 | 4.40 | 10.06 | -5.67 | [-5.90, -5.43] | [-6.05, -5.30] | [-8.06, -3.50] | YES |

negative delta = the method is better than B1 (EER is an error rate).

## 2. Gates

- **G17 (N5, ESN)**: `ESN_FAIL` - ESN-LR at m=5 13.27 vs the best control OC+B1 7.76 (delta 5.51, (b) interval [4.231945025882896, 6.794665527272493]).
- **G18 (N6, projection)**: `PROJ_FAIL` - PROJ-ind strict_draw 14.47 vs B1 14.45 (delta 0.02, (b) interval [-0.018825364972594124, 0.06187870655338064], seen delta 0.07).

Tracked flags (U1', v3): `SPEAKER_GAP` = True for ['B9']; `LP_GAIN_IS_SPEAKER_LOCAL` = False (B9 minus B1 on unseen = -1.19, (b) [-2.1206788684265963, -0.2320875955407337], old [-10.264579860558005, 4.670090701425648]).

### U10 levers (backbone and fusion, no method change)

| variant | seen | strict_draw |
|---|---|---|
| ENS3-S3 | 8.92 | 13.41 |
| ENS3-S5 | 8.92 | 13.42 |
| ENS3-mix | 8.92 | 13.42 |
| mms-1b|S3 | 8.90 | 13.53 |
| mms-1b|S5 | 8.88 | 13.52 |
| xls-r-1b|S3 | 10.90 | 15.29 |
| xls-r-1b|S5 | 10.90 | 15.29 |
| xls-r-2b|S3 | 10.19 | 14.45 |
| xls-r-2b|S5 | 10.19 | 14.45 |

| contrast | mask | delta | (b) interval | clear of 0 |
|---|---|---|---|---|
| ENS3-S3 - mms-1b|S3 | seen | 0.02 | [-0.20, 0.16] | no |
| xls-r-2b|S5 - xls-r-2b|S3 | seen | 0.00 | [-0.00, 0.01] | no |
| ENS3-S3 - xls-r-2b|S3 | seen | -1.27 | [-1.44, -1.20] | yes |
| mms-1b|S5 - xls-r-2b|S3 | seen | -1.31 | [-1.58, -1.08] | yes |
| xls-r-2b|S5 - xls-r-2b|S3 | strict_draw | 0.00 | [-0.00, 0.00] | no |
| ENS3-S3 - xls-r-2b|S3 | strict_draw | -1.03 | [-1.41, -0.87] | yes |
| ENS3-S3 - mms-1b|S3 | strict_draw | -0.12 | [-0.51, 0.60] | no |
| mms-1b|S5 - xls-r-2b|S3 | strict_draw | -0.93 | [-1.82, -0.48] | yes |

Reading: the zero-shot column (S5) is worth ~0.00 points everywhere; swapping the feature block from `xls-r-2b` to `mms-1b` is worth more than any method in this batch, and the 3-backbone ensemble only reproduces the best single member.

## 3. U1' v2 -> v3

pooled EER, v2 dev (68 speakers) -> v3 dev (122 speakers):

| method | seen v2 | seen v3 | unseen v2 | unseen v3 | strict_draw v2 | strict_draw v3 |
|---|---|---|---|---|---|---|
| B1 | 7.27 | 10.19 | 12.52 | 11.96 | 15.23 | 14.45 |
| B3 | 7.45 | 11.80 | 13.51 | 17.37 | 16.40 | 17.15 |
| B7 | 7.01 | 9.70 | 12.37 | 13.88 | 15.66 | 15.13 |
| B9 | 5.58 | 7.25 | 12.45 | 10.77 | 18.62 | 15.04 |
| C1 | 7.45 | 10.13 | 13.95 | 14.84 | 16.23 | 16.10 |

On **identical v2 draws**, the old interval rule and the new (a) rule disagree in **10 of 12** cells: the old rule keeps 0 inside an interval whose mean-difference interval excludes it.

## 4. Fact statement

**YES** - on unseen speakers the best method is OC+B1@m=10 at 6.19 pooled EER versus B1 at 10.51, i.e. a -4.32-point difference with the (b) interval [-4.97, -3.22], so at least one method is >= 2.0 points better than B1.

On `strict_draw` the best method is OC+B1@m=10: delta -8.60 points, (b) interval [-10.23, -7.34], clears the 2.0-point bar.

## 5. Files behind this page

- `results/u1p_speaker_split_v3.csv`
- `results/u1p_intervals.csv`
- `results/u1p_v2v3.csv`
- `results/u1p_v2_rule_compare.csv`
- `results/u8_esn.csv`
- `results/u8_intervals.csv`
- `results/u8_gate.json`
- `results/u9_projection.csv`
- `results/u9_intervals.csv`
- `results/u9_gate.json`
- `results/split_v3_audit.csv`
- `splits/speaker_split_v3.json`
