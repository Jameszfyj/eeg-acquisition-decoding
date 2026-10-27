# MI Studio v2 — prepare 锚点取窗 + 六类源模型（全数据）

2026-10-27。v2 是把 v1（`mi_studio/`）的取窗口径换掉的**新包**（`mi_studio_v2/`），v1 与远端旧训练器
一行未改。v1 的两个 profile（`platform_replay_v1`、`l100d250h4_v1`）在 v2 里原样保留，老结果仍可复现。

## 1 口径（本次唯一的规则变化）

- 一只试次的窗口支持段：**`[anchor − P, task_end + post]`**，`anchor = prepare`（目标提示出现），
  `P = pre_seconds = 1.0 s`（因果 FIR 历史），`post = 1.0 s`。
- 锚点取数：优先 `events_json` 里 `kind=="prepare"` 的 `sample_index`；缺失时依次回退
  `trials_json.preview_sample`（在线数据）→ `collection.prepare_seconds`（会话元数据）→ profile 默认 2.0 s。
  每次回退都在 `bank.meta["anchor_kinds"]` 里计数，报告不隐瞒。
- 锚点吸附到最近的 250 Hz 列（prepare 的 `sample_index` 带设备时钟 ±10 ms 抖动）；吸附超过 50 ms 判
  `anchor_off_grid` 拒该试次，不静默平移。
- 有效段 `[anchor, task_end + post]` 内，按 **0.5 s 步长**切 2 s 窗（`n_time = 500 @ 250 Hz`）。
- 预处理不变：每连续段 L100D250H4 因果滤波（201 tap Kaiser 5.65，4–100 Hz @1000 Hz → 抽 4 → 4 Hz 高通 @250 Hz），
  每段开头丢 1.0 s；列时刻 = 该列最早原始样点号 / 1000（设备时钟，不补群延迟）；无陷波。
- 与 v1 的差别：v1 在 `cue + 0.6/1.0/1.4` 固定切 3 窗，且 `[cue−1 s]` 只是历史；v2 的窗从
  **文字出现那一刻**起铺，cue 前的 2 s（本数据实测 prepare lead 2.39–2.41 s，6 s 协议 2.49–2.51 s）
  全部可用，每试次 11 窗（T=4 s）/ 15–16 窗（T=6 s）。

## 2 数据池（24 份会话 / 1591 试次 / 21 254 窗）

| 来源 | 会话 | 试次 | 窗口 | 锚点来源 |
|---|---|---|---|---|
| 10-21 cargame dy | 6 | 386 | 5 789 | `prepare_seconds`（日志头，2.0 s） |
| 10-23 collection | 3 | 358 | 3 938 | `prepare` 事件（2.4 s） |
| 10-24 collection | 3 | 270 | 2 970 | `prepare` 事件（2.4 s） |
| 10-26 collection（6 s 协议） | 5 | 500 | 7 659 | `prepare` 事件（2.5 s） |
| 10-25 校准 + 带提示 | 7 | 77 | 898 | `preview_sample` 43 / `profile_fallback` 39 |
| **合计** | **24** | **1 591** | **21 254** | `prepare` 1 129 · `prepare_seconds` 386 · `preview_sample` 43 · `profile_fallback` 39 |

逐类（输出列顺序 `[左旋臂, 左握拳, 右握拳, 右旋臂, 双脚, 舌头]`）：294 / 296 / 312 / 316 / 137 / 236 试次。
10-26 的 id 4（双手、双手旋臂）与 cargame 的抓握码（1/2/5）在本模型没有输出列，按 `unused` 明确排除
（`rejected.unmapped_label = 160`）。10-21 的 `20261021_153552` 只有抓握码，整份不入池。

## 3 训练（device 3，RTX 3080 Ti）

| 项 | 值 |
|---|---|
| 配方 | IFNetRTG-L width 16、`d = 8`、SVM 初始化、200 / 30 / 200 轮、patience 40、batch 128 |
| 参数 | 2 782 |
| 评估 | **无留出折**（`eval = none`）：全部 24 份会话都进训练池 |
| 拟合 | 1 353 试次拟合 / 238 试次（15%，按类别分层）只用于选轮次，不参与任何权重更新 |
| 用时 | 339 s（预训练 74 轮早停，best epoch 34，inner-valid bACC 0.779） |
| 产物 | `model.pt`（34.6 KB）+ `report.json` |

没有留出准确率是这次配置的**已知代价**：模型是全量拟合件，不是可引用的泛化数字。要数字就跑
`train.eval = "holdout"`（留出 10-25 的 6 份会话，6 折）。

## 4 用法

```python
import sys; sys.path.insert(0, r"F:\project\机械手2026")
from mi_studio_v2.training import SourceModel

m = SourceModel.load(r"F:\project\机械手2026\mi_studio_v2\models\six_class_pre_v2_allsess_s0.pt")
logits = m.predict_logits(X)     # X: (N, 30, 500) float32 µV，按 v2 规则切的 2 s 窗
proba  = m.predict_proba(X)      # (N, 6)，列序 = [左旋臂, 左握拳, 右握拳, 右旋臂, 双脚, 舌头]
```

切窗（新数据）：

```python
from mi_studio_v2.preprocess import build_bank
bank = build_bank([{"path": r"D:\新会话目录",
                    "mapping": {"rows": [{"id": 0, "target": 1}, {"id": 1, "target": 2},
                                         {"id": 2, "target": 0}, {"id": 3, "target": 3},
                                         {"id": 4, "target": 4}, {"id": 5, "target": 5}],
                                "many_to_one_confirmed": False},
                    "_outputs": m.outputs}],
                  "l100d250h4_pre_v2", require_all_outputs=False)
```

## 5 复现

```bash
# 生成任务配置（逐会话产出精确映射）
python -B -X utf8 mi_studio_v2/jobs/make_jobs.py
# 窗口库（低内存：X 落 X.npy，其余 meta.npz；24 份约 90 s，1.3 GB）
python -B -X utf8 -m mi_studio_v2 run --config mi_studio_v2/jobs/job_bank_v2.json
# 训练（远端；或本地把 bank 指到 runs/mi_studio_v2/bank_v2）
python -B -X utf8 -m mi_studio_v2 train --config mi_studio_v2/jobs/job_train_v2_remote.json
```

远端位置：代码 `/root/autodl-fs/jixieshou/mi_studio_v2/code/`、窗口库 `…/bank_v2/`、
产物 `…/runs/train_src4_seed0/`（device 3 的共享盘，其它设备同样可见）。

## 6 已知限制

- 10-25 的校准会话没有 prepare 标记，锚点按 2.0 s 回退；带提示会话的 preview lead 实测 0.83–1.01 s，
  与 collection 的 2.4 s 不同 —— 这批窗的**位置**与 collection 不一致（`anchor_kinds` 可筛）。
- 6 份 cargame 的锚点来自日志头的 `prepare_seconds`，不是硬件标记，属于重建而非实测。
- `rejected` 里 `anchor_off_grid 6 / beyond_post_window 4 / start_not_aligned 2 / trial_crosses_gap 1`
  都是被显式拒掉的窗/试次，没有静默补齐。
