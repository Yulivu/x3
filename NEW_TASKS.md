# X3 · NEW_TASKS 01（hongyu 追加；拷到 `x3/NEW_TASKS.md`）

> 给执行 agent。**优先级覆盖**：本文件里标 `[URGENT]` 的任务，在你做完手头的工作包后立即开始，优先于所有尚未开始的 P1/P2/P3；W5/W6 的 `final` 重跑与 G0（`mms-1b`、`xls-r-1b` 上传和特征）继续并行。旧判据 N1 / N2 / N4 及其结果（`C1_FAIL`、`PAST_FAIL`、`DECONF_NOT_SHOWN`）**原样保留在报告里，不删除、不改动**；新判据另存 `x3/claims_v2.yaml`。仍然：设计决定只用 dev 区域，没有 `x3/FREEZE_OK` 不评估 test。我（任务书作者）没有在本机运行过任何代码，规格以你的单测为准。

## 0. 为什么追加（来自你上一次的检查点报告）
1. 骨干换对了（零样本 20.21 / AUC 0.863；LR 加 290 标签 7.70 / 8.76）。
2. 多层级探针 C1、PAST、去混杂机制都没过门；**格先验这条路是死的**：Line A 的 13.64 属于"290 条直接训头"路线，先验校正路线只有 25.10，oracle 格先验也只有 21.52，B2 反而比 B1 差 4.85 个点。
3. 目前最强的是**基线 B9 标签传播**（6.08 / 7.33）；CASA 赢均匀分配 6.42 个点，但对"全局熵"只赢 0.23（区间跨 0）。
4. 标签集中在 20 个说话人时所有方法一起崩（B1 8.68 → 12.70）：**说话人覆盖比格覆盖重要**。
5. 所有主表数字是 dev 区域的 `clips` 口径——评测 clip 的说话人几乎都在 L 里出现过；**严格（未见说话人）口径还没有数**。B11（MLP）26.6% 明显不对。

下面的任务就是围绕这五条：先验证数字是否站得住（U1–U3），再把方法重心改到"**主动选标 + 图传播**"（U4–U5），并更新预登记判据（U7）。

## 1. claims_v2（预登记：在运行 U4–U6 之前写入 `x3/claims_v2.yaml`，把 sha256 写入 `x3/claims_v2.sha256`，此后不得修改）
| 判据 | 内容 |
|---|---|
| **N1′ 图传播融合** | M-ALP 的学习器（LR+LP 融合）对 `max(B1, B9)`：k=290，在 `clips` 与 `unseen` 两个口径上，pooled **或**平台内宏平均低 ≥0.5，抽样级配对 95% 区间排除 0；在两个骨干上都成立 |
| **N2′ 选标器** | M-ALP 选标器在 k=100 的 pooled 比 U（每格均匀）低 ≥3.0，**且**比 {E, K, TypiClust-lite} 中最强者低 ≥0.5（区间排除 0）。只赢 U 不赢后者 → 记 `SEL_ONLY_VS_UNIFORM`，不算新意 |
| **N3′ 说话人覆盖** | 同预算 k=100，SC（说话人覆盖约束）比 CC（格覆盖约束）在 `clips` 口径 pooled 低 ≥1.0；`unseen` 口径只报告 |
| **N4′ 标签效率** | M-ALP 在 k=100 的 pooled 与 B1+U 在 k=290 的差 ≤0.3（非劣）→ 标签减少 2.9 倍 |
| **S-F′ 少样本 SOTA** | k∈{100, 290}：M-ALP 对 BASE_F′ 的**每一个**基线，三个口径（pooled / 平台内宏平均 / 最差平台）中至少两个更低且配对区间排除 0，第三个不差于 +0.3；`clips` 与 `unseen` 两个口径都要成立；两个骨干。BASE_F′ = {B1, B2, B3, B4, B5, B7, B8, B9, B10, B11′} |
| **S-Z、S-R** | 不变 |

## 2. 任务

### U1 [URGENT] · 严格口径与 seen/unseen 说话人分解（CPU，≤1.5h，dev 区域）
- 骨干 `xls-r-2b|S3`（第二骨干就绪后补跑）。方法：B1、B9、B7、B8、C1、B2、B3；k=290 和 k=100。
- 三种口径：
  - **`seen`**：E 中说话人在 L 里出现过的 clip；
  - **`unseen`**：E 中说话人在 L 里从未出现的 clip（支持不足的平台标 NaN，只报 pooled）；
  - **`strict_draw`**（另一种抽样）：把 dev 说话人随机分成两半 S_L 与 S_E，L 只从 S_L 抽（每格最多 10 条，格内不足取一半），E = S_E 的全部 clip。
- 每种 100 次抽样；输出 `results/u1_speaker_split.csv`（schema 同 X3 第 9 节，加列 `mask ∈ {seen, unseen, strict_draw}`、`n_speakers_in_L`、`n_labeled`）。**主表里必须同时给出实际的 `n_labeled`。**
- 判定：某方法 `unseen` 的 pooled ≥1.5× `seen` → `SPEAKER_GAP`；B9 在 `unseen` 上不优于 B1（配对区间含 0 或更差）→ `LP_GAIN_IS_SPEAKER_LOCAL`。

### U2 [URGENT] · 基线完整性（CPU，≤1h）
- **B11（MLP）26.62% 不对**：先在合成数据上与 B1 对拍（线性可分数据上 MLP 应≈LR），检查标准化、学习率、训练步数、早停、权重衰减、类别比；修好后用同一 16 组网格重跑，报告前后。修不好 → 标 `B11_BROKEN`，改用 sklearn `MLPClassifier(early_stopping=True)` 作为 B11′ 并说明。
- **B0-LineA-probe**：W4 里 13.64 所属的"290 条直接训头"路线（`COMPARISON.md` 第 42 行），在新骨干特征上的版本；与 B1 对照，二者应接近，不接近就解释差异。
- **B2-oracle**：用 dev 全部真值算格先验的上界，复现 21.52 这个量级。
- **B2 为什么比 B1 差**：验证是"重复计数"（探针已含格信息，又叠加先验）——对比 B2 与"B1 + 仅全局偏移"。
- 汇总 `results/u2_baseline_audit.csv`。

### U3 [URGENT] · 骨干预训练数据的身份重叠审计（CPU，≤1h）
- AntiDeepfake 的训练数据含 VoxCeleb2 等名人语料，ML-ITW 是 180 位公众人物，**可能有身份重叠**。取 VoxCeleb1 / VoxCeleb2 的公开元数据文件（`vox1_meta.csv`、`vox2_meta.csv`，以 VGG 官方页为准，体量很小）；规范化 ML-ITW 的 `speaker` 字段（小写、去标点、变音符归一），做精确匹配和模糊匹配（token-sort ratio ≥90，人工抽检 20 对）；报告重叠说话人数和其 clip 占比。
- 把 dev 说话人分成 `overlap` / `non_overlap`，对 B1、B9、零样本 AntiDeepfake 分别报告 pooled（支持够时加平台内宏平均）。重叠比例 ≥20% **或**两组差 ≥3 个点 → `IDENTITY_OVERLAP_FLAG`。
- 元数据拿不到 → `UNAVAILABLE`，**不要为此耽误别的**。

### U4 [URGENT] · M-ALP：主动标签传播（CPU，≤4h，dev 区域）
**学习器 L-GRAPH**
- 图：PCA-256 空间，余弦 kNN，`k_g ∈ {10,15,30}`；边权 `exp(sim/T)`，T = 相似度中位数；对称化；可选 speaker-aware 加权：同一说话人的边权 ×`w_s`，`w_s ∈ {1,2}`。
- 传播：Label Spreading `F = (I−αS)^{-1}Y`，`α ∈ {0.8,0.9,0.99}`；加类质量归一化（Zhu 2003），使预测的假占比等于 L 的全局假率。
- 融合：`score = logit_LR + η·F_resid`，`η ∈ {0, 0.25, 0.5, 1}`（η=0 即 B1，η 很大即 B9）。
- 超参 `(k_g, α, w_s, η)` 共 ≤16 组，用 L 上的**留一节点**对数损失选（把一个已标节点的标签藏起来再预测）。
**选标器**（预算 k）
- 热启动：PCA 空间 k-means（n0=60 个簇），每簇取离中心最近的 clip 标注；
- 之后分 4 批，每批 `(k−60)/4`：`score_i = u_i · ρ_i · c_i`，其中 `u_i = 1−|2q_i−1|` 为当前融合模型的不确定度，`ρ_i` 为局部密度（kNN 平均相似度），`c_i = 1−exp(−d_i/τ)` 为到最近已标节点的覆盖项（`τ` 取该距离的中位数）；贪心取并去冗余（余弦距离 <0.1 视为重复）；
- 变体 **SC**（说话人覆盖）：在 clip 数 ≥30 的说话人里，先保证每人至少 1 个已标 clip，之后再按 score 选；变体 **CC**（格覆盖）：同样约束但对"平台×语言"格。
**评测**
- k ∈ {30, 100, 290, 600}（dev 按实际可用调整，记录 `n_labeled`）；口径 `clips` + `seen/unseen`（U1）；k=290 做 100 次抽样，其余 50 次。
- 对比：B1+U、B9+U、CASA+LR、E+LR、K+LR、TypiClust-lite+LR（簇内典型点）、M-ALP(SC)、M-ALP(CC)、M-ALP。
**消融矩阵**（k=100 与 290，各 30 次）：选标器 {U, E, K, DWU（密度加权不确定度）, TypiClust-lite, SC-DWU, CC-DWU, full} × 学习器 {LR, LP, LR+LP}。输出每个因子的边际效应（配对区间）。**目的：找出真正起作用的成分**，别再用"格感知"去解释增益。
**单测**：合成两簇数据（簇内含说话人子簇）上，LP+DWU 在小预算下比均匀选标好 ≥2 个点；kNN 图对称，归一化后谱半径 ≤1。
**输出**：`results/u4_*.csv`（schema 同第 9 节）。

### U5 · 标签噪声与小预算稳健性（CPU，≤1h）
对 M-ALP、B9、B1、C1：标签翻转 5% / 10% / 20%，k ∈ {30, 100, 290}，各 30 次。C1 在 10% 噪声下崩到 17.95，要看到 M-ALP 的曲线。输出 `results/u5_noise.csv`。

### U6 · 继续原计划
W8（Track Z）与 W15（ITW）排在 U1–U4 之后继续；Track Z 加 **Z5**：用零样本 logit 初始化，k=0 时只做伪标签传播（标签无关）；ITW 上也要跑 M-ALP（格 = 说话人，预算 k=290，每说话人 5 条）。

### U7 · 判据 v2 的自动判定（`decide.py` 增加，门只记录）
- **G12（N1′）**：`MALP_PASS` / `MALP_FAIL`。
- **G13（N2′）**：`SEL_PASS` / `SEL_ONLY_VS_UNIFORM` / `SEL_FAIL`。
- **G14（N3′）**：`SPEAKER_COVERAGE_PASS` / `SPEAKER_COVERAGE_FAIL`；另报 `unseen` 口径的差。
- **G15（N4′）**：`LABEL_EFFICIENCY_PASS` / `LABEL_EFFICIENCY_FAIL`。
- **G16（S-F′）**：`SOTA_F2_DEV_YES` / `SOTA_F2_DEV_NO`。
- `DECISION` 行追加 `G12 … G16` 与 U1–U3 的旗标（`SPEAKER_GAP`、`LP_GAIN_IS_SPEAKER_LOCAL`、`IDENTITY_OVERLAP_FLAG`、`B11_BROKEN`）。

## 3. 顺序、时间与汇报
- U1、U2、U3 并行（CPU）先做，约 2 小时后更新一次 `CHECKPOINT_REPORT.md`，**并把 U1 的 `seen / unseen / strict_draw` 主表置顶**；然后 U4 → U5 → U6 → U7。
- U4 完成后再更新一次 `CHECKPOINT_REPORT.md`，列出 G12–G16 与消融矩阵的边际效应。
- 仍然：不碰 test；调参对等（每个方法 ≤16 组、L 内 CV 或留一节点）；转导式方法（B6、B7、B8、B9、M-ALP、Z 系列）在报告里逐一标注；不要因为某个门没过就改阈值。
- GitHub 推送被本地代理挡住时，继续用 bundle 同步，并把 bundle 路径和 md5 写进 `progress.log`。
