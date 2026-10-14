# X3 · 长跑 goal：AntiDeepfake 骨干 → 少样本适配方法（MPP / PAST / CASA）→ 对强基线的 SOTA 证据

> 给执行 agent。机器：32GB vGPU 箱。**队列的串行工作量约 30 小时，两条泳道并行约 18–20 小时墙钟；不自动关机**——队列做完后进入"等待模式"，只有 hongyu 创建 `x3/SHUTDOWN_OK` 才关机。全自动，不等确认，某项失败记录后继续。凡写了"断言/单测/门"的地方都必须落成脚本。我（任务书作者）**没有在本机运行过任何代码**，下面的代码片段只给思路，以你的自检为准。

## 0. 一页纸

### 0.1 目标与成功标准
hongyu 的 ML-ITW 项目（下称 Line A）要在 WWW 2027 赛道 "Web Mining, Multimedia and Multilingual Content Analysis" 做出：**(1) 赛道契合；(2) SOTA；(3) 有新意的技术贡献**，而且不是审计型论文。Line A 现状：冻结 XLS-R 特征 + 290 个目标域标签（29 个"平台×语言"格 × 每格 10 条）训练的小探针，加格先验校正；头条 pooled EER 13.64%、平台内宏平均 14.56%、最差平台（TikTok）20.79%（clips 口径，3 次独立抽样的均值）。

本 goal 的产出是下面这些**预登记的判据**（写进 `x3/claims.yaml`，由 `decide.py` 在 dev 区域上计算；最终在 test 区域上重算，需要 `x3/FREEZE_OK`）：

| 判据 | 内容 |
|---|---|
| **S-F** 少样本 SOTA | k=290 时，终版方法 M* 对基线集合 BASE_F 里的**每一个**基线：三个口径（pooled / 平台内宏平均 / 最差平台）中至少两个更低且抽样级配对 95% 区间排除 0，第三个不差于 +0.3 个点。BASE_F = {B1 LR, B2 LR+格偏移(Line A), B3 LR+每平台偏移, B4 GP, B7 朴素伪标签, B8 FixMatch-lite, B9 标签传播, B10 FT-last-block, B11 MLP 探针}。在**零样本最好的两个骨干**上都要成立 |
| **S-Z** 零标签 SOTA | k=0 时，pooled ≤24.65（比论文最好的 27.65 低 ≥3 点）且比最好的 AntiDeepfake 零样本再低 ≥2 点，平台内宏平均不差于零样本 |
| **S-K** 预算曲线 | k∈{30,100,290,600,1200} 全部不差于最强基线 |
| **S-R** 复刻 | ITW 上排序不被推翻（见 G9） |
| **N1** | C1（多层级探针）通过 G4 |
| **N2** | PAST（先验感知自训练）通过 G6 |
| **N3** | CASA（格感知主动选标）通过 G7 |
| **N4** | 机制证据通过 G10（去混杂） |

**新意的处理规则**：N1–N4 中 ≥2 个通过，才把方法当"主贡献"；只通过 N1，记为"实现细节"；一个都不通过，报告里写"方法增益不足"，保留基线与协议结论，**不要为了过门调任何东西**。

### 0.2 运行方式
- tmux 会话 `x3-main`，编排器 `x3/run_queue.py` + `x3/queue.yaml`（见 3 节）。可恢复：已完成的工作包（输出文件齐全 + `DONE` 标记）自动跳过。
- **控制文件（由 hongyu 创建，你只读）**：`x3/PAUSE`（存在则做完手头的工作包后空转）、`x3/FREEZE_OK`（允许进入 test 区域的最终评测，见 12 节）、`x3/SHUTDOWN_OK`（允许关机）、`x3/NEW_TASKS.md`（hongyu 追加的任务，按最低优先级排进队列）。
- 队列全部完成后写 `x3/ALL_DONE`，进入**等待模式**：每 30 分钟写一行心跳，继续做 P3 积压，轮询上面的控制文件。**没有 `SHUTDOWN_OK` 就不关机。**
- 每 30 分钟往 `x3/progress.log` 写一行（`时间 | 当前工作包 | GPU 利用率与显存 | 磁盘剩余 | 一句话进展`）；每个工作包结束 `git commit` + `git push`；每 3 小时和每个层级结束时自动生成 `x3/CHECKPOINT_REPORT.md`（当前 `DECISION` 行 + 关键表），hongyu 随时可以看。
- 没有墙钟硬上限、没有 GPU 小时上限。**GPU 有活时不得空转超过 20 分钟**；GPU 没活时释放显存但保持进程存活。

### 0.3 队列总览（串行估时；泳道 G=GPU/网络，C=CPU）

| WP | 层级 | 泳道 | 依赖 | 估时 | 内容 |
|---|---|---|---|---|---|
| W0 | P0 | C | — | 0.3h | 读 Line A 协议与代码，复述；盘点 |
| W1 | P0 | G | — | ≤1.5h | AntiDeepfake 环境 + 权重 |
| W2 | P0 | G | W1 | ≤3.5h | 4 个骨干逐层特征 + 零样本分数 + 零样本表 |
| W3 | P0 | C | W0 | 0.3h | 说话人 30/70 划分（冻结）+ 文件名元数据 |
| W4 | P0 | C | W0,W3 | 2h | Line A 原特征 100 次抽样复跑 + 对账 + 机制表 |
| W5 | P0 | C | W2,W3 | 2h | 新骨干 spec 搜索 + 100 次 dev 抽样 + 骨干排名（G3） |
| W6 | P0 | C | W3（临时）/G3（定稿） | 3h | 基线全集 B1–B11 + C1(MPP) @k=290 |
| W7 | P0 | C | W6 | 1h | 预算曲线 k∈{30,100,290,600,1200} |
| W8 | P0 | C | W5 | 1.5h | Track Z 零标签（Z0–Z3） |
| W9 | P0 | C | W6 | 1h | 泄漏与负对照审计 |
| W10 | P1 | G+C | G3 | 2.5h | 增强视图特征（Fam-A×2、Fam-B×1） |
| W11 | P1 | C | W6,W10 | 3h | PAST 全版 + 消融 |
| W12 | P1 | C | W6 | 2h | CASA 全版 + 策略对比 |
| W13 | P1 | G | G3 | 3–4h | FT-last-block 基线（B10） |
| W14 | P1 | C | W6 | 1.5h | 标签传播与 GraphSSL 变体 |
| W15 | P1 | G+C | G3 | 2.2h | ITW 特征 + 少样本复刻 |
| W16 | P1 | C | W11,W12 | 2h | 因子消融矩阵 + 去混杂机制证据 |
| W17 | P2 | G | G3 | 2h | NDA 变体 + 额外骨干 |
| W18 | P2 | C | W10,W11 | 1.5h | C4：集成与测试时增强 |
| W19 | P2 | C | W10,W11 | 1h | 未见转码链鲁棒性 |
| W20 | P2 | C | W11,W12 | 1.5h | 超参与设定敏感性 |
| W21 | P2 | C | W6 | ≤2h | 尝试官方 ADD-GP 代码（可放弃） |
| W22 | P3 | G/C | — | 积压 | 抽样 100→200、更多种子、更多 k、Z4（标签无关的自训练）等 |
| WF | F | G+C | FREEZE_OK | 4–6h | test 区域最终评测 + 论文资产（见 12 节） |

### 0.4 优先级与追加规则
- 顺序：P0 → P1 → P2 → P3；同层级内按依赖和泳道空闲度调度。**CPU 泳道不得空等 GPU**：W6/W7/W9 先在**原 XLS-R 特征**上开发并产出 `provisional` 结果，G3 出来后在选定的（骨干, spec）上重跑为 `final`。
- 落后时的割舍阶梯（每次舍弃写进 REPORT 的"偏离"）：W22 → W21 → W20 → W19 → W18 → W17（先 NDA 后额外骨干）→ W16 的 TTA 因子 → W14 → W15 的少样本复刻（保留零样本）。**永不舍弃**：W0–W9、W10–W13 中的 W11/W12、对账表与机制表、零样本表、`DECISION` 行。
- **回退**：W1 在 45 分钟内装不上 fairseq → 用 `transformers` 的基础 SSL（`facebook/wav2vec2-xls-r-2b`、`wav2vec2-xls-r-1b`、`facebook/mms-1b`）提特征（无零样本分数），标 `ANTIDEEPFAKE_UNAVAILABLE`；W8 的 Z 系列改为只做元数据和 EM 部分。

## 1. 已核实的事实（2026-10-15；核不上的写进"偏离"）

### 1.1 AntiDeepfake（NII Yamagishi 实验室）
- 论文 arXiv 2506.21090（v4）；代码 `github.com/nii-yamagishilab/AntiDeepfake`（BSD-3）；**权重 CC BY-NC-SA 4.0（仅研究用途，不要上传权重或衍生模型）**。训练数据：56,370 小时真 + 18,280 小时假，100 多种语言，含 MLS、FLEURS、MLAAD、SpoofCeleb、VoxCeleb2、VoxCeleb2-Vocoded 等。零样本 ITW EER 2.25%（wav2vec-large NDA）、Deepfake-Eval-2024 30.06%。
- HF 仓库 ID（各有默认版与 `-nda` 版）：`nii-yamagishilab/xls-r-2b-anti-deepfake`、`xls-r-1b-anti-deepfake`、`mms-1b-anti-deepfake`、`mms-300m-anti-deepfake`、`hubert-xlarge-anti-deepfake`、`wav2vec-large-anti-deepfake`、`wav2vec-small-anti-deepfake`（`-nda` 同名加后缀）。
- **下载不需要申请**（10-15 核过）：HF 仓库 `gated=false`、`private=false`，文件为 `config.json` + `model.safetensors`（`xls-r-2b` 约 8.7GB）。Zenodo 记录 `15580543` 访问权限为 **Open**，总计 46.3GB：`xlsr_2b.ckpt` 8.7GB、`xlsr_1b.ckpt` 3.9GB、`mms_1b.ckpt` 3.9GB、`mms_300m.ckpt` 1.3GB、`hubert_xl.ckpt` 3.9GB、`w2v_large.ckpt` 1.3GB、`w2v_small.ckpt` 0.38GB（各有 `_nda` 版），下载地址 `https://zenodo.org/records/15580543/files/<文件名>?download=1`，页面上每个文件有 MD5。官方 `main.py inference --ckpt_path` 吃的是 Zenodo 的 `.ckpt`。
- **安装**（GitHub README）：`conda create --name antideepfake python==3.9.0`；`conda install pip==24.0`；`pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118`；fairseq 用固定提交 `git clone https://github.com/pytorch/fairseq && git checkout 862efab86f649c04ea31545ce28d13c59560113d && pip install --editable .`（README 注明"最新提交不行"）；`pip install speechbrain==1.0.2`；`pip install tensorboard tensorboardX soundfile pandarallel scikit-learn numpy==1.21.2 pandas==1.4.3 scipy==1.7.2`。仓库自带 `install.sh`，**以它为准**。fairseq 与 speechbrain 放在 AntiDeepfake 目录**旁边**。（HF 模型卡写的是 "fairseq 0.12.2"，两者不一致时以 GitHub README 为准。）
- **推理**：官方入口 `python main.py inference hparams/<名>.yaml --base_path ... --exp_name ... --test_csv ... --ckpt_path ...`，输出 `evaluation_score.csv`（real 类概率，阈值 0.5）。HF 模型卡里还有一份推理脚本：输入转单声道、重采样 16kHz、**对波形做 layer norm**，对最后一层帧特征做**平均池化**，过线性头，输出 real/fake 概率。**类别顺序以官方脚本为准**（我没核实 fake/real 的下标）。
- **README 没有写怎么取中间层特征**。fairseq 的 `Wav2Vec2Model.extract_features(source, padding_mask, mask=False)` 返回字典，其 `layer_results` 是各层输出的列表（每项第一个元素形状 T×B×C；对 pre-LN 模型，它是最终 layer norm 之前的值）。具体以你读到的版本为准，写进 `x3/src/extract_layers.py`，并说明下标约定（第 0 个隐状态 = 输入嵌入还是第 1 层输出）。

### 1.2 ML-ITW、ITW 与 Line A
- ML-ITW：HF `nccm2p2/ML-ITW`，24,529 条，7 个平台，14 种语言，180 位说话人；**无官方划分、无少样本协议**；论文 v3 里多个基线的 ACC 约 26%、F1 约 41，恰好是"全判为假"的水平（全假：ACC 25.9%、F1 41.2），**所以论文的 27.65% 不是强基线**。平台 × 标签计数见 X2 的 1.1 节。
- ITW（第二数据集）：HF `mueller91/In-The-Wild`，`gated=false`，单个 `release_in_the_wild.zip` 约 8GB；58 位公众人物，约 20.8h 真 + 17.2h 假，16kHz；HF 标签 cc-by-sa-4.0，Fraunhofer 页写 Apache-2.0，两处不一致，报告里照实写。
- Line A（hongyu 提供）：平台内宏平均 14.56%（clips），最差平台 TikTok 20.79%（clips），见 `docs/COMPARISON.md` §E.5；**每个模型种子（0/1/2）各自独立重抽一份 290 标签**（每格 10 条，29 格，双掩码），头条是这三次的均值±标准差；`fsc_var.py` 是参数化的抽样循环（目前 8 次抽样 × 3 种子），改 `draws` 即可到 100 次，约慢 12 倍。
- **待你核实、我没有看到**：29 个格的过滤规则、标签抽取是否限制说话人、clips / strict 两个掩码的精确定义、探针结构与超参、特征用的哪一层和池化方式。W0 要**逐条复述**。
- 已知文献与基线（我只读了摘要，引用前请核对）：ADD-GP（arXiv 2505.23619，代码 `github.com/NetaGlazer/ADD-GP`，冻结 XLS-R-2B + Dirichlet GP）；AAAI 2026 标量 logit 偏移校准（arXiv 2602.01973，图像）；2606.21584（保序变换不改 EER；伪标签校准在 96% 为假的数据集上退化到 50% HTER；建议报迁移阈值 HTER）；提示调优 2410.09869；元学习 LoRA 2502.10838。

## 2. 硬规则
1. **不改**已冻结的协议表（`docs/COMPARISON.md` §0.0–§0.2）和所有已有结果文件；新增文件只放 `x3/`；分支 `x3/main`。
2. **设计决定只用 dev 区域**（W3）。test 区域在拿到 `x3/FREEZE_OK` 之前**不评估**，也不得用 test 说话人的任何标签做决定。
3. 评分一律用 **logit（或 logit 差）**，不用 softmax 概率。EER 约定：spoof 为正类、分数越大越像 spoof、全部相同返回 50.0。沿用 Line A 的 EER 与掩码实现（不改它），另写三个口径函数并加单测：与 `sklearn.metrics.roc_curve` 版 `eer()` 在随机数据上差 <0.1；全相同 =50；分数取反 =100−EER。
4. 三个口径缺一不可：**pooled EER、平台内宏平均、最差平台**（该平台两类各 ≥30 条才算，否则 NaN；宏平均对有效平台等权）；另报 AUC 和迁移阈值下的 HTER（阈值取 L 上的 EER 点，用到评测区域）。
5. 抽样种子：第 d 次抽样用 `20261015 + d`；每次抽样的标签 clip id 存 `x3/draws/{protocol}_{d}.json`；**同一次抽样里所有方法共用同一个 L 与同一个评测集 E**。
6. **调参对等**：每个方法（含全部基线）的超参搜索都不超过 16 组配置、都只在 L 内用分层 5 折 CV 选、用同样的随机折。把每个方法的搜索网格写进 `x3/configs.yaml`，报告里列出。
7. 磁盘：先 `df -h`。本任务产物预计 ≤120GB（特征 ≤40GB，增强视图临时音频 ≤10GB，权重 ≤40GB）。**不要删除任何旧数据**。剩余磁盘 <20GB 时暂停新的特征抽取，先删除未被选中骨干的权重（保留特征），仍不够就写进 `progress.log` 并等待。
8. GPU：先 `nvidia-smi`。若 tmux 里有 `probe-*`（X1/X2）在占用 GPU（>4GB），等它们结束，或把批大小减半，不要抢。
9. 网络按之前的做法（HF 用 `HF_ENDPOINT=https://hf-mirror.com`；GitHub 不通用 AutoDL 学术加速）。
10. 任何单步卡住 >30 分钟：记录现象，重试一次，仍失败则标 `FAILED` 并继续不依赖它的工作包。
11. 版权与伦理：ML-ITW 的音频含公众人物声音，**不进 git，不复制到别处**；AntiDeepfake 权重不上传。
12. **所有"选哪个"的决定都由脚本按门的规则做出**（`decide.py`），不等人。
13. 所有新模块先写单测再使用；每个工作包的输出文件原子写入（先写临时文件再改名）。
14. **一切需要申请的数据都不碰**：Deepfake-Eval-2024（需申请，条款只准评测，也不要用 `xxuan-speech` 的转存版绕过）、`jungjee/spoofceleb`（人工审批）、`mueller91/MLAAD`（需同意条款）。本任务用到的数据与权重都不需要申请：ML-ITW、AntiDeepfake、ITW。

## 3. 目录、接口与编排器
```
x3/
  TASKBOOK.md  T0.txt  ENV_READY  claims.yaml  configs.yaml  queue.yaml  run_queue.py
  PAUSE FREEZE_OK SHUTDOWN_OK NEW_TASKS.md ALL_DONE      # 控制文件，只有 hongyu 创建前四个
  src/  extract_layers.py metrics3.py draws.py probe_zoo.py baselines.py mpp.py em_prior.py past.py alloc.py graph_ssl.py ft_last.py views.py zlab.py audit.py
  scripts/  w0_*.py ... wf_*.py  decide.py  make_report.py
  tests/  test_metrics3.py test_batch_vs_single.py test_baselines.py test_mpp_synth.py test_em_synth.py test_past_synth.py test_alloc.py test_views.py
  results/  logs/  splits/  draws/  gates.json  chosen_backbone.json  progress.log  REPORT.md  CHECKPOINT_REPORT.md
$DATA/x3/feats/<backbone>/  mean.npy std.npy layers.json scores.csv clip_order.csv DONE
$DATA/x3/views/<famA1|famA2|famB1>/<backbone>/  spec_feats.npy scores.csv DONE
```
- `mean.npy`：float16，`[N, L+1, D]`（所有隐状态的帧均值，不含填充帧）。`std.npy`：float16，`[N, Ls, D]`，只存 `layers.json` 里的 `STD_LAYERS`（约 8 层：`⌊0.25L⌋, ⌊0.5L⌋, ⌊0.625L⌋, ⌊0.75L⌋, ⌊0.875L⌋, L−1, L` 及 Line A 的同位层）。`clip_order.csv` 与 Line A manifest 的 `clip_id` 逐行一致（断言）。`scores.csv`：`clip_id, logit_fake_minus_real, p_real`。
- **编排器 `run_queue.py`**：读 `queue.yaml`（每个工作包：`id, tier, lane, deps, est_h, cmd, outputs`），按依赖和泳道调度；每个工作包完成后写 `<outputs>/DONE`；失败重试一次后标 `FAILED`；每次循环检查 `PAUSE` / `FREEZE_OK` / `SHUTDOWN_OK` / `NEW_TASKS.md`；写心跳。`queue.yaml` 里 W0–W21 的依赖按 0.3 节表。

## 4. P0 工作包

### W0 · 复述与盘点（0.3h）
1. 记录开机时刻到 `x3/T0.txt`；把本文拷成 `x3/TASKBOOK.md`。
2. 读 `docs/COMPARISON.md` §0.0–§0.2、§E.5 与 `fsc_var.py`（及其调用的函数）。在 `x3/REPORT.md` 开头**逐条复述**第 1.2 节"待你核实"列出的六件事，每条附代码位置（文件:行号）。
3. 盘点：Line A 的 manifest 与特征缓存路径/形状、已有结果、音频路径；`df -h`、`nvidia-smi`、`nproc`、`free -g`。
4. 复核 platform × label 计数与 X2 的 1.1 节表一致。

### W1 · 环境与权重（≤1.5h；若 `x3/ENV_READY` 已存在且冒烟通过，则只做冒烟）
1. 建环境 `antideepfake`，按 1.1 节安装。45 分钟内装不上 fairseq：记录报错，按"回退"处理。
2. 下载权重（默认版，按优先级）：`xls-r-2b` → `mms-1b` → `xls-r-1b` → `hubert-xlarge`。**首选 HF（hf-mirror）**；HF 慢或失败时改用 Zenodo 直链并用页面 MD5 校验。记录大小与校验值。不要对同一个骨干下两种格式，唯一例外：保真自检若选用官方 `main.py inference`，则只为 `xlsr_2b` 额外下载一份 Zenodo 的 `.ckpt`。
3. 冒烟：加载 `xls-r-2b-anti-deepfake`，对 1 条样本跑通；打印隐状态个数（预期 49）、形状、最终层池化后的 real/fake 概率；确认 fake/real 下标写进 `layers.json`；成功后创建 `x3/ENV_READY`。

### W2 · 特征与零样本分数（GPU；单骨干 ≤50 分钟，超时记 `SKIPPED_TIME` 并继续下一个）
对每个骨干（按优先级）：
1. 音频策略**与 Line A 完全一致**（读它的提特征脚本；记录长度/裁剪/归一化）；再按官方做法对波形做 layer norm。
2. 按长度排序成动态批（总样本数 ≤160 秒/批），**用 padding mask**，帧均值/标准差只对有效帧计算。fp16。
3. 每个隐状态取帧均值存 `mean.npy`；`STD_LAYERS` 取帧标准差存 `std.npy`。零样本分数：最终层帧平均池化 → 模型自带线性头 → `logit_fake − logit_real`，存 `scores.csv`。
4. **保真自检**（`tests/test_batch_vs_single.py`）：随机 20 条，批处理与单条处理的池化特征余弦 ≥0.999、零样本分数最大绝对差 <1e-2；再随机 50 条，与官方参考实现的 real 概率比较（二选一，选能跑通的：HF 模型卡里的推理脚本，或官方 `main.py inference`）：Spearman ≥0.99、最大绝对差 <0.05。不满足则以官方预处理为准重做。
5. 零样本表 `results/zeroshot_mlitw.csv`：每个骨干的 pooled EER、平台内宏平均、最差平台（及名称）、AUC、HTER@p=0.5、7 个平台各自 EER、各语言 EER（每类 ≥30 才算）。另附"元数据查表"一行作参照（沿用 Line A 的 28.77/30.41）。
6. 每个骨干抽完立即写 `feats/<backbone>/DONE`，供 W5 流水线使用。

### W3 · 划分与元数据（0.3h，W0 后立即做）
1. 说话人不相交的 30/70 划分（组 = 说话人，按 label 分层的带约束随机分配）。约束：每个平台、每个类别，dev ≥30 条、test ≥100 条；不满足换种子重来（≤200 次），记录最终种子。写 `splits/speaker_split_v2.json`（种子、dev/test 说话人、每个平台×类别计数表、文件 md5）。**冻结后不再改**。
2. 解析 manifest 里的 `audio_name`/路径：是否含源视频 ID 或片段序号？若有一条解析规则覆盖 ≥95% 的条目，新增 `video_id` 列，报告视频数、每视频条数、说话人与视频的包含关系（只报告）。

### W4 · Line A 原特征上 100 次抽样复跑（CPU）
1. **复现**：按原设置（3 种子）跑 `fsc_var.py`，要求 pooled / 平台内宏平均 / 最差平台与已存结果的差 ≤0.5 个点。差更大：记录并继续。
2. **100 次抽样**：`draws=100`，protocol 标记 `v1_full`（整池、clips 与 strict 两个掩码），三个方法：**M0** 探针单独（不加格先验）；**M1** 仅格先验（BB+EB，分数 = logit(π̂_cell)）；**M2** Line A（探针 + 先验，原样）。
3. 输出逐抽样 CSV 与汇总（均值、标准差、2.5%/97.5% 分位）。**对账表** `results/reconciliation.csv`：3 次抽样的头条 vs 100 次抽样的均值；**机制表**：M0/M1/M2 的 pooled、平台内宏平均、最差平台。
4. **异常核对**：M2 的 clips 与 strict 两个 EER 是否逐位相同（头条里两者都是 13.64）。相同则找原因，写入报告，不要自行改 Line A 的代码。

### W5 · 新骨干：spec 搜索、100 次抽样、骨干排名（每个骨干抽完就开始）
1. **dev 区域**：只保留 dev 说话人的 clip，抽样仍是"每格最多 10 条"（格内可用 clip 不足 `2×需要数` 的格取 `⌊n/2⌋`，<4 的格丢弃；记录实际标签数）。掩码只用 `clips`。
2. **spec 菜单**（每个骨干，L = 层数）：S1 = Line A 同位层与同池化的类比；S2 = 最后一层均值+标准差；S3 = 第 `⌊0.75L⌋` 层均值+标准差；S4 = 第 `⌊0.5L⌋`、`⌊0.75L⌋`、`L` 三层均值拼接；S5 = S3 再拼上零样本 logit。探针类型与超参选择沿用 Line A 的做法；若 Line A 是固定超参，则对新维度另加"L 内 5 折 CV 选 C 的 LR"变体（M2-LR）并同时报告。
3. 对每个（骨干, spec）跑 30 次 dev 抽样的 M2，按**平台内宏平均**（平局看 pooled）选出每个骨干的最优 spec；再对最优 spec 跑 100 次 dev 抽样的 M0/M1/M2。原 XLS-R 特征用同样的抽样作为配对对照。
4. 排名表 `results/backbone_rank.csv`：每个（骨干, spec）的 M2 平台内宏平均、pooled、最差平台（均值±标准差）及相对原特征的配对差（抽样级，均值与 95% 区间）。
5. 全部骨干完成后 `decide.py` 给出 **G3**：选出的（骨干, spec）写进 `chosen_backbone.json`，同时记录**第二名骨干**（S-F 要求两个骨干）。

### W6 · 基线全集 + C1（CPU，dev 区域；先在原特征上开发，G3 后重跑 `final`）
- **输入与预处理**：特征 z_i = 选定 spec 的池化向量；在 dev 区域**全部 clip（无标签）**上拟合标准化与 PCA（保留 256 维或 95% 方差，取小者），所有方法共用。L 与 W5 相同（每格最多 10 条）；评测集 E = dev 区域去掉 L。
- **基线与方法**（规格见 7 节）：B1 LR；B2 LR+格偏移（Line A 的 M2）；B3 LR+每平台偏移；B4 GP；B5 kNN/原型；B6 零样本+无标签 EM；B7 朴素伪标签；B8 FixMatch-lite（原特征上的弱/强扰动用高斯噪声+特征 dropout 近似；W10 之后有真正的增强视图版本）；B9 标签传播；B11 MLP 探针；**C1 多层级探针（MPP）**及其变体 MPP-MLP、MPP-noInt、MPP-platformOnly。B10（FT-last-block）在 W13。
- **抽样与指标**：k=290（每格 10）× 100 次；输出 `results/w6_draws_<backbone>_<spec>.csv`、`results/w6_summary.csv`、`results/w6_paired.csv`（C1 对 B1/B2/B3/B11 的抽样级配对差）。
- **单测**：B1 与 sklearn `LogisticRegression` 的 logit 相关 ≥0.999；`test_mpp_synth.py`：合成 d=20、29 格、格效应 `N(0,0.5²)`、N=6000、290 标签，C1 恢复的格效应与真值相关 ≥0.6 且 AUC ≥ B1；`test_em_synth.py`：合成数据上 EM 恢复格先验，MAE ≤0.1。
- **G4**：C1 相对 {B1,B2,B3,B11} 中最强者（按平台内宏平均）：pooled 低 ≥1.0 **或** 平台内低 ≥0.7，抽样级配对区间排除 0 → `C1_PASS`，否则 `C1_FAIL`。
- **G5**：B7 在"极端先验格"（真实 fake 率 ≥0.8 或 ≤0.1，用 dev 区域真值算）里伪标签假占比与真值偏差 >20 个点的格占比 ≥30% → `PL_COLLAPSE`，否则 `PL_OK`（只记录，用于支撑 PAST 的动机）。

### W7 · 预算曲线（CPU，1h）
每格标签数 m ∈ {1, 3, 10, 20, 40}（对应 k≈30、100、290、600、1200；dev 区域内每格 `m_c ≤ ⌊n_c/2⌋`），每个 k 做 50 次抽样，方法 B1、B2、B3、B11、C1（以及 W11/W12 完成后的 PAST、CASA）。输出 `results/w7_curve.csv`；**S-K** 的数据来源。

### W8 · Track Z 零标签（CPU，1.5h，dev 区域）
k=0，只用元数据和无标签音频；评测集 = dev 区域全部。
- **Z0**：AntiDeepfake 零样本分数 s_i（每个骨干）。
- **Z1**：每平台标准化 `s'_i = (s_i − med_p)/IQR_p`（中位数/四分位距取自该平台的无标签 clip）。
- **Z2**：带格混合权重的两成分高斯混合：类条件参数 (μ0,σ0,μ1,σ1) 全局共享，格先验 π_c 自由，EM 20 步；初值用零样本分数的符号划分；分数 = `log N(s|μ1,σ1) − log N(s|μ0,σ0) + logit π_c`（π_c 向平台、全局收缩，κ=10）。
- **Z3**：Z2 + 对 W10 的增强视图取平均（W10 之后补跑）。
- 输出 `results/w8_zlabel.csv`（三个口径）。**G8**：Z2 或 Z3 的 pooled ≤24.65，且比最好的零样本再低 ≥2 点，平台内宏平均不差于零样本 → `Z_PASS`，否则 `Z_FAIL`（只记录）。

### W9 · 泄漏与负对照审计（CPU，1h）
在 W6 的最终配置上，各 30 次抽样：
1. 标签置换（在平台内打乱 y）：所有方法 EER 应≈50（±3），否则有泄漏或实现错误。
2. 格元数据置换（打乱 cell id）：C1 的增益应消失（回到 B1 水平 ±0.5）。
3. strict 掩码（排除已标说话人）下的结果 vs clips 掩码；有 `video_id` 时再加视频不相交掩码。
4. 近重复检查：评测 clip 与已标 clip 的最近邻余弦 >0.99 的占比。
5. 元数据-only 分类器（平台×语言 → LightGBM）的 EER，作为"捷径水平"参照。
输出 `results/w9_audit.csv`；任一项异常写 `AUDIT_FLAG` 并在报告置顶。

## 5. P1 工作包

### W10 · 增强视图特征（GPU+CPU，2.5h；仅选定骨干，G3 之后）
- **转码链生成**（ffmpeg，16kHz 单声道 WAV 输入输出，长度偏差 ≤50ms，否则补/截）：
  - **Fam-A（训练期视图，用于 PAST 的一致性与 TTA）**：A1：Opus（`libopus`），码率 ∈ {12,16,24,32,48} kbps；A2：AAC（`aac`），码率 ∈ {24,32,48,64,96} kbps。每个视图再以概率施加：随机增益 ±6dB（p=0.5）、粉噪声 SNR 10–30dB（p=0.3）、降采样到 8kHz 再升回（p=0.3）。
  - **Fam-B（留出的评测视图，用于 W19 的"未见链"）**：B1：双重转码 Opus 24kbps → MP3（`libmp3lame`）64kbps，再叠 8kHz 限带（p=0.5）。
  - 每个视图每条 clip 的随机参数由 `seed=20261015+clip_id` 决定，存 `views/<名>/params.csv`。
- 对选定骨干提取每个视图的**选定 spec 向量**与零样本分数，存 `views/<名>/<backbone>/spec_feats.npy`、`scores.csv`。临时音频提完即删（保留 md5 与参数）。
- 单测 `test_views.py`：每个视图形状正确、无 NaN、时长偏差 ≤50ms；Fam-A 与 Fam-B 的参数集不相交。

### W11 · PAST 全版 + 消融（CPU，3h，dev 区域）
- 规格见 7.3。抽样 k=290 × 50 次（dev）。对比：C1、B7（朴素）、B8（FixMatch-lite 真视图版）、PAST。
- **消融**：PAST w/o 先验感知（全局阈值）、w/o 强视图一致性、w/o EM（固定 π 来自 L）、w/o 配比、仅 1 轮。
- **G6**：PAST 对 C1 平台内宏平均低 ≥1.5 **或** pooled 低 ≥1.0，抽样级配对区间排除 0 → `PAST_PASS`，否则 `PAST_FAIL`。
- 单测 `test_past_synth.py`：合成数据（含极端先验格）上，朴素伪标签的格先验偏差 >0.2，PAST 的偏差 ≤0.1。

### W12 · CASA 全版 + 策略对比（CPU，2h，dev 区域）
- 规格见 7.4。k=100 和 290，各策略 30 次抽样：U 每格均匀、P 按格大小比例、N Neyman、E 全局熵、K 格内 k-center、BADGE-lite（LR 梯度嵌入 + k-means++）、**CASA**。
- **G7**：CASA 在 k=100 的 pooled 比 U 低 ≥1.0，抽样级配对区间排除 0 → `CASA_PASS`，否则 `CASA_FAIL`。

### W13 · FT-last-block 基线 B10（GPU，3–4h）
- 目的：给"同样 290 个标签直接微调骨干"一个公平基线。
- 做法：对选定骨干，缓存 dev 区域所有 clip 的第 `L−1` 层输出（ragged fp16，`[T_i, D]` + 偏移表；xls-r-2b 约 6GB），用 fairseq 的最后一层模块 + 最终 LayerNorm + 平均池化 + 自带线性头在缓存上微调（AdamW，lr∈{1e-5,3e-5} 由 L 内 CV 选，5 个 epoch，批 16，不用类别权重），评测在 E 上。每次抽样约 1 分钟量级，k=290 × 50 次抽样。
- 若 1.5 小时内搭不起来：退而求其次，只微调"最终 LN + 线性头"（等价于带 LN 的 LR），并在报告里标 `B10_REDUCED`。

### W14 · 标签传播与 GraphSSL（CPU，1.5h）
- B9：在 PCA 空间建 kNN 图（k=15，余弦，边权 `exp(sim)`），标签传播（α=0.9，20 次迭代），初始 Y0 = L 上的 ±1、其余 0。
- **GraphSSL**：`score = C1 logit + η·Y_prop`，η∈{0.25,0.5,1} 在 dev 上选（作为设计决定），k=290 × 50 次。输出同 schema。仅在同一 dev 区域上做（无标签图包含 E），报告里注明是**转导式**。

### W15 · ITW 特征与少样本复刻（GPU+CPU，2.2h）
- 下载 `mueller91/In-The-Wild` 的 `release_in_the_wild.zip`，解压，清点条数/说话人。对选定骨干（和第二名）提取特征与零样本分数。
- 零样本：EER（与 AntiDeepfake 官方 NDA wav2vec-large 的 2.25% 作量级参照，不要求一致）。
- **少样本复刻**：格 = 说话人（58 个），标签预算 k=290（每说话人 5 条），评测 = 其余 clip；方法 B1、B2、B3、B11、C1、PAST（无增强视图版）、CASA；每个 50 次抽样；口径：pooled、"说话人内宏平均"（替代平台内）、最差说话人（该说话人两类各 ≥30 才算）。
- **G9**：C1 在 ITW 上不比 B1 差超过 0.3 个点，且 PAST 不比 C1 差 → `ITW_REPLICATES`，否则 `ITW_MIXED`。**ITW 不做 dev/test 划分，所有设计决定来自 ML-ITW dev，一次性运行。**

### W16 · 因子消融矩阵与机制证据（CPU，2h，dev 区域）
- **因子矩阵**：C1 开/关（关=B1+后置偏移）、PAST 开/关、CASA 开/关（关=U），2³ 共 8 格，k=100 与 290，各 30 次抽样；输出每个因子的边际效应（均值与配对区间）与交互项。
- **去混杂机制（G10）**：对每次抽样，取"检测器部分" `d(z)=wᵀz`（B1 的 w vs C1 的 w），计算 d 预测平台的一对多宏平均 AUC；C1 的平台可预测性比 B1 低 ≥0.03，且平台内 EER 不变差 → `DECONF_OK`，否则 `DECONF_NOT_SHOWN`。
- 输出 `results/w16_factorial.csv`、`results/w16_mechanism.csv`。

## 6. P2、P3 工作包

### W17 · NDA 变体与额外骨干（GPU，2h）
对零样本最好的两个骨干跑 `-nda` 版；再抽 `mms-300m`、`wav2vec-large`（用于集成）。同 W2 的格式与自检。结果并入 `backbone_rank.csv`（若某 NDA/额外骨干在 dev 上的 M2 平台内宏平均比当前选定者低 ≥0.5 且区间排除 0，则 G3 自动更新并**重跑 W6 的 `final`**，旧结果保留为 `superseded`）。

### W18 · C4：集成与测试时增强（CPU，1.5h，dev 区域）
- E1：top-2 骨干的校准 logit 平均；E2：对 Fam-A 视图与干净视图的 logit 取平均（TTA）；E3 = E1+E2。对 C1 与 PAST 各做一遍。k=290 × 50 次。

### W19 · 未见转码链鲁棒性（CPU，1h）
在 Fam-B 视图上评测（训练只见过 Fam-A）：方法 B1、B2、C1、PAST、PAST+TTA；指标 `ΔEER = EER(Fam-B) − EER(干净)`（三个口径）。这是"平台转码链"这一 Web 故事的直接证据。

### W20 · 敏感性（CPU，1.5h）
PCA 维数 {64,128,256,512}；σ 网格；格定义变体（仅平台 / 仅语言 / 平台+语言可加 / 平台×语言）；标签噪声 5%、10%；标签抽取方式（每格均匀 / 全局随机 / 说话人聚集：所有标签来自 20 个说话人）。每项 30 次抽样，只跑 B1、B2、C1、PAST。

### W21 · 尝试官方 ADD-GP（≤2h，可放弃）
克隆 `https://github.com/NetaGlazer/ADD-GP`，读其 README；若 ≤2h 内能把我们的特征/标签接入其 Dirichlet-GP 分类器，则作为 B4' 加入基线；否则保留 sklearn-GPC 类比，并在报告里写明"B4 不等同 ADD-GP"。

### W22 · P3 积压（按顺序，做到有人叫停）
1. 抽样 100→200（W6、W11、W12 的头条配置）；2. 更多 k 点；3. **Z4**：标签无关的 PAST（用 Z2 的后验做伪标签，训练 LR，迭代）；4. 其他 spec/层的敏感性；5. 对第二名骨干重复全部 P1 实验；6. 更多转码链家族。

## 7. 方法规格

### 7.0 公共记号
cell c=(p,ℓ)；L 为标签集，U 为无标签集（含 E）；z 为预处理后的特征；`σ(x)=1/(1+e^{−x})`；π_c 为格 c 的 fake 率。

### 7.1 基线
- **B1 LR**：L2 逻辑回归（不用类别权重），C∈{1e-3,1e-2,1e-1,1,10}，L 内分层 5 折 CV 按对数似然选；HTER 阈值取 L 上的 EER 点。
- **B2 LR+格偏移**：Line A 的 M2（T4A 已有）；若接口不通用，则类比版：B1 的 logit + `logit(π̂_cell) − logit(π̂_L)`，π̂ 用 BB+EB（沿用 Line A 实现）。
- **B3 LR+每平台偏移**（AAAI 2026 后验校准类比）：先训 B1，固定 w，仅拟合每平台一个标量 a_p，带 `N(0,σ²)` 先验（σ∈{0.25,0.5,1,2} 由 CV 选）。
- **B4 GP**：sklearn `GaussianProcessClassifier`（RBF，PCA 前 64 维），ADD-GP 的简化类比。
- **B5 kNN/原型**：余弦 k=15 距离加权，Platt 校准（L 内 CV）。
- **B6 零样本+无标签 EM**：零样本 logit s 用 L 做 Platt 校准 `a·s+b`；在 U 上按格做 Saerens-EM 估 π_c（向平台、全局收缩）；分数 = `a·s+b + logit(π_c) − logit(π_global)`。
- **B7 朴素伪标签**：从 B1 出发，3 轮，每轮取全局置信度 ≥τ（τ∈{0.8,0.9,0.95} 由 CV 选）的无标签 clip 作伪标签（不控制类别比例），与 L 合并重训（伪标签权重 0.5）。
- **B8 FixMatch-lite**：弱视图 = 干净特征，强视图 = Fam-A 视图特征（W10 之前用"高斯噪声 + 特征 dropout 0.2"近似）；置信度 ≥τ 的弱视图预测作为强视图的硬标签，权重 0.5，3 轮。
- **B9 标签传播**：见 W14。
- **B10 FT-last-block**：见 W13。
- **B11 MLP 探针**：1 个隐层 256，GELU，dropout 0.3，weight decay 1e-2，Adam lr 1e-3，全批量 100 步，早停由 L 内 CV 决定；格偏移同 B2（BB+EB）。

### 7.2 C1 多层级探针（MPP）
```
η_i = wᵀz_i + μ + α_{p(i)} + β_{ℓ(i)} + γ_{c(i)}
α_p, β_ℓ ~ N(0, σ_p²)   γ_c ~ N(0, σ_γ²)   w ~ N(0, 1/λ)   μ 不惩罚
最小化 −Σ_{i∈L}[y_i η_i − log(1+e^{η_i})] + (λ/2)‖w‖² + Σ α²/(2σ_p²) + Σ β²/(2σ_p²) + Σ γ²/(2σ_γ²)
```
- 超参网格（共 16 组，与基线对等）：`(σ_p, σ_γ) ∈ {0.25,0.5,1,2}²`，λ 取 B1 在 CV 里选出的 C 的倒数；全部由 L 内分层 5 折 CV（按对数似然）选。
- 优化：PyTorch L-BFGS 或 scipy；L 中没出现的格只用 α+β；输出 logit（可校准的对数几率）。
- 变体：MPP-MLP（`wᵀz` 换成 B11 的 MLP）、MPP-noInt（去掉 γ）、MPP-platformOnly（只留 α）。
- **预期**：联合训练使 w 只需解释格内变异，减少与格先验的重复计数；这是 G4 与 G10 要检验的假设。

### 7.3 C2 PAST（先验感知自训练）
输入：C1 模型、无标签集 U、干净特征 z_i 与 Fam-A 视图特征 z_i^{(a)}（a=1,2）。
- **检测器 LLR**：`d_i = wᵀz_i + μ − logit(π̄_L)`（π̄_L 为 L 的全局 fake 率）；用 L 上的温度 T 校准（`d ← d/T`，T 由 L 内 CV 最小化对数损失）。
- 每轮 r=1..R（R=4）：
  1. **EM 估格先验**：`q_i = σ(d_i + logit π_c)`；`π_c ← (Σ_{i∈U_c} q_i + Σ_{i∈L_c} y_i + κ·π_{p(c)}) / (|U_c| + |L_c| + κ)`，κ∈{5,10,20}（CV 选），平台先验 π_p 为其格的加权平均；迭代 20 次或收敛。
  2. **按先验配比选伪标签**：每格 c，配额比例 `ρ_r = 0.1·r`；假类取 `round(ρ_r|U_c|π_c)` 个最高 q_i，真类取 `round(ρ_r|U_c|(1−π_c))` 个最低 q_i；置信度阈值 `τ_r = 0.75+0.05r`；无标签数 <10 的格跳过。
  3. **强视图一致性**：对被选中的伪标签 clip，同时把它的 Fam-A 视图特征加入训练（伪标签来自干净视图，权重 0.25）；另对所有无标签 clip 加对称 KL 一致性项 `λ_u·KL(q(z_i) ‖ q(z_i^{(a)}))`（`λ_u=0.1`，实现为额外的软标签样本，权重 0.1）。
  4. 在 `L ∪ 伪标签（权重 0.5）∪ 强视图样本` 上重拟合 C1（超参每轮仅在 L 内重选）。
- 输出最终分数（含最终 π_c 的格先验）。
- **与 B7 的区别（论文要强调的）**：B7 用全局阈值、不控制类别比例，在极端先验格（Douyin≈86% 假、YouTube≈8% 假）会塌向多数类；PAST 的配额由 EM 估计的格先验决定，且以 L 为锚。

### 7.4 C3 CASA（格感知主动选标）
- 热启动：`n0=60`（约每格 2 条）随机均匀；拟合 C1。
- 之后分 4 批，每批 `B=(k−60)/4`：对每个无标签 clip i（格 c）打分：`score_i = H(q_i) + ρ·V_c·(N_c/N)`，其中 `H` 为 Bernoulli 熵，`V_c = ab/((a+b)²(a+b+1))`，`a=1+pos_c`、`b=1+neg_c`（当前格内标签计数），`N_c` 为格大小；ρ∈{0.5,1,2} 在 dev 上选（设计决定）。
- 选择：按分数降序贪心取，每格上限 `≤2×平均`，并用格内最远点（PCA 空间余弦距离 <0.1 视为重复）去冗余；每批后重拟合 C1。
- 对照策略：U、P（∝N_c）、N（∝N_c√(π̂_c(1−π̂_c))，π̂ 来自热启动）、E（全局熵）、K（格内 k-center）、BADGE-lite。

### 7.5 Track Z / C4
见 W8、W18。

## 8. 统计、claims 与自动决策
- **抽样级配对**：同一次抽样里各方法共用 L 与 E，差值按抽样配对；报告均值差、2.5%/97.5% 分位（对 100 次抽样的经验分布）、胜率。最终 test 报告另给**说话人聚类 bootstrap** 区间（每次抽样内 B=200）。
- **多重比较**：`claims.yaml` 里预登记的比较列表固定；对列表内的比较做 Holm 校正后报告 p 值（只作辅助，判据以配对区间为准）。
- `claims.yaml` 含：S-F、S-Z、S-K、S-R、N1–N4 的精确定义（0.1 节）、BASE_F 列表、阈值。`decide.py` 在 dev 上计算 `SOTA_F_dev`、`SOTA_Z_dev` 等，并在 `FREEZE_OK` 后对 test 重算。
- **门**：

| 门 | 判据 | 结果 |
|---|---|---|
| G0 特征健康 | ≥3 个 AntiDeepfake 骨干完成特征与零样本分数，且保真自检通过 | `pass`；1–2 个=`amber`；0 个=`fail`（回退并标注） |
| G1 余量 | 零样本最好的 AntiDeepfake 骨干：pooled >15 **或** 平台内宏平均 >20 | `HEADROOM_OK`；否则 `HEADROOM_SMALL` |
| G2 复现与稳定性 | W4 复现差 ≤0.5；100 次抽样均值与 3 次头条的 pooled 差 ≤2 | `REPRODUCED/NOT_REPRODUCED`；`STABLE/OPTIMISTIC` |
| G3 骨干选择 | dev 上平台内宏平均最低的（骨干, spec）；相对原特征的配对差 ≥0.5 个点且区间排除 0 才换 | 骨干 ID 或 `KEEP_ORIGINAL`（并给出第二名） |
| G4–G10 | 见 W6、W11、W12、W8、W15、W16 | 各自的 PASS/FAIL |
| G11 | `SOTA_F_dev`：dev 上 S-F 是否成立（两个骨干） | `SOTA_F_DEV_YES/NO` |
| 诊断 | W4 的 clips 与 strict 是否逐位相同；W9 的 `AUDIT_FLAG` | 原样报告 |

- `decide.py` 打印一行：`DECISION: G0=… G1=… G2=…/… G3=… G4=… G5=… G6=… G7=… G8=… G9=… G10=… G11=…`，并写 `gates.json`、`chosen_backbone.json`；未做的写 `NOT_RUN`。**今晚没有 KILL**：门只记录、只决定后续用哪个配置。

## 9. 结果文件 schema（进 git 的部分）
- `results/*draws*.csv`：`protocol, region, backbone, spec, method, k, draw, mask, n_labeled, n_eval, pooled_eer, within_macro_eer, worst_platform, worst_platform_eer, auc, hter_pooled` + 7 个 `eer_<平台>`（NaN 表示支持不足）。ITW 用 `eer_spk_*` 汇总代替。
- `results/summary_*.csv`、`results/*_paired.csv`、`results/zeroshot_mlitw.csv`、`results/reconciliation.csv`、`results/backbone_rank.csv`、`results/w7_curve.csv`、`results/w8_zlabel.csv`、`results/w9_audit.csv`、`results/w16_*.csv`。
- `splits/speaker_split_v2.json`、`draws/*.json`、`gates.json`、`chosen_backbone.json`、`claims.yaml`、`configs.yaml`、`queue.yaml`、`MD5SUMS`、`logs/*`。
- 不进 git：`$DATA/x3/` 下的特征、视图、权重、音频。

## 10. `REPORT.md` 与 `CHECKPOINT_REPORT.md`（脚本从 csv/json 生成，只陈述事实，不写"有前景"之类的判断）
1. 摘要：`DECISION` 行；S-F / S-Z / S-K / S-R / N1–N4 当前状态；被割舍的项；耗时。
2. W0 六件事复述；W2 零样本表与自检；W3 划分；W4 对账表与机制表；W5 骨干排名。
3. W6 主表（k=290，两个骨干，所有方法）；W7 预算曲线数据；W8 Track Z；W9 审计。
4. W11 PAST 与消融；W12 策略对比；W13 B10；W14；W15 ITW；W16 因子矩阵与机制。
5. W17–W21；偏离与异常（现象、处理、影响）；复现命令与 commit。

## 11. 陷阱
1. 批处理里的填充帧不能参与均值/标准差；fairseq 的 `padding_mask` 要传对；`extract_features` 的返回形状可能是 T×B×C，注意转置；层下标约定写清楚。
2. 零样本分数用 logit 差，不要用已饱和的概率；EER 在并列分数处要稳定。
3. 多进程读 `mean.npy` 用 `np.load(mmap_mode='r')`。
4. dev 区域抽样时，格内可用 clip 数少于需要数要显式处理并记录实际标签数。
5. 标准化与 PCA 只用区域内的**无标签**特征拟合；超参只在 L 内做交叉验证；**所有方法同一折划分**。
6. 转导式方法（B6、B7、B8、B9、PAST、GraphSSL、Z 系列）的无标签集合就是评测区域本身——这是转导式设定，报告里逐一标注；不得把 L 的标签泄漏到 U 的特征预处理里。
7. 不要用 test 说话人的任何标签做决定；`FREEZE_OK` 之前不评估 test。
8. 权重许可 CC BY-NC-SA：不上传权重或衍生模型。
9. C1 的格效应在 L 中某些格只有极少样本时不稳，先验 σ 必须由 CV 选，不要手调。
10. EM 的 π_c 要裁剪在 [0.02, 0.98]，避免 logit 爆炸。
11. 增强视图的随机参数必须与干净视图一一对应保存，保证可复现；Fam-B 的参数集不得出现在任何训练中。
12. 对基线和方法**同样努力地调参**（硬规则 6）；报告里列出每个方法的搜索网格与选中的值分布。
13. GPU 任务和 CPU 任务争核时，GPU 泳道至少留 4 核。

## 12. 最终评测（需 `x3/FREEZE_OK`）与收尾

### 12.1 WF · 最终评测
没有 `FREEZE_OK` 之前，这一节**一律不执行**。hongyu 创建 `FREEZE_OK` 后：
1. **WF1 冻结**：把 dev 上的选择写成 `x3/final_config.json`（骨干与 spec、各方法的网格、C1/PAST/CASA 是否纳入——按 G4/G6/G7 的结果自动决定、PCA 维数、η、ρ 等），计算其 sha256，写入 `final_config.sha256`。此后不得修改。
2. **WF2 test 区域评测**：100 次抽样 × clips 与 strict 两个掩码，k=290；方法 = BASE_F 全部 + C1 + PAST + CASA + C4；k 曲线（k∈{30,100,290,600,1200}，各 50 次）；Track Z（Z0–Z3）；两个骨干。**test 只跑这一次**。strict 掩码下支持不足的平台标 NaN。
3. **WF3 ITW**：用冻结配置一次性运行（若 W15 已在 ITW 上跑过设计无关的配置，则只补终版方法）。
4. **WF4 论文资产**：主表（三口径）、k 曲线、因子矩阵、去混杂机制图的数据表、Fam-B 鲁棒性表、`claims_evidence.csv`（每条预登记判据 → 对应的表格单元与结论 `HOLDS/FAILS`）、局限清单（脚本能检出的：小平台支持不足、转导式设定、骨干预训练数据可能含这批公众人物）。

### 12.2 收尾与等待模式
1. 队列全部完成后写 `x3/ALL_DONE`，最后一次 `git add x3 && git commit && git push origin x3/main`（附 `gates.json`、`REPORT.md`、`CHECKPOINT_REPORT.md`、`MD5SUMS`）。
2. 汇报要点：`DECISION` 行 + S-F/S-Z/S-K/S-R/N1–N4 状态 + 主表（k=290）+ 偏离清单（≤15 行）。
3. 进入等待模式：每 30 分钟心跳，继续 P3 积压，轮询 `PAUSE` / `FREEZE_OK` / `SHUTDOWN_OK` / `NEW_TASKS.md`。**只有看到 `x3/SHUTDOWN_OK` 才关机**（AutoDL 上执行 `shutdown`）；若 `tmux ls` 里还有其他 `probe-*` 会话在这台机器上跑，则不关机，只汇报。
