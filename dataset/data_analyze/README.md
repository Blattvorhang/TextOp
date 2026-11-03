# 数据分析工具

脚本按用途分类；命令从仓库根目录运行。现有分析结果继续放在 `analysis_output/`，包括人工复核通过的 20 条 T-pose NPZ。

| 子目录 | 用途 | 主要入口 |
| --- | --- | --- |
| `goal_distribution/` | Root / end-effector goal 位置、朝向及片段分布 | `analyze_64f_goal_xy.py`、`analyze_64f_end_effector_xyz.py`；对应 `visualize_*_viser.py` |
| `action_distribution/` | 动作类别、短动作、文本描述分布 | `analyze_action_distribution.py`、`analyze_short_actions.py`、`analyze_text_description_similarity.py` |
| `dataset_statistics/` | 数据集总时长、帧数及 BONES-SEED 综合分析 | `compare_dataset_duration.py`、`analyze_hiphi_duration.py`、`count_bonesseed_duration.py`、`analyze_bonesseed.py` |
| `pose_space/` | 旋转表示、动作姿态流形和曲面 | `analyze_rpy.py`、`visualize_motion_manifold.py`、`visualize_motion_surface.py` |
| `planner_diagnostics/` | 规划衔接和恢复曲线示意 | `pnc_gap.py`、`time_decayed_recovery_*.py` |
| `tpose/` | HiPhi T-pose 自适应清洗、端点时长分析及历史基线 | 正式入口 `apply_adaptive_endpoint_trim.py`；测量入口 `analyze_endpoint_trim_duration.py` |
| `visualization/` | 通用动作 NPZ 播放，显式文件清单、路径切换与区间标记 | `motion_viser.py` |
| `self_collision/` | G1 训练数据自碰撞扫描、拟剔除量估算 | `analyze_self_collision.py`、`estimate_self_collision_removal.py` |

## 数据集时长占比

```bash
MPLCONFIGDIR=/tmp/textop-duration-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/dataset_statistics/precompute_duration_statistics.py
MPLCONFIGDIR=/tmp/textop-duration-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/dataset_statistics/compare_dataset_duration.py
```

预计算脚本先按 `filter_and_copy_bones_data.py` 的默认关键词筛选 BONES-SEED CSV 对应的 PKL 名称，并保存逐文件清单 `analysis_output/dataset_statistics/duration_comparison/bones_filter_manifest.json`；同时读取 HiPhi 的 `full_scan_endpoints_adaptive/removal_intervals.json` 与 `hiphi_self_collision_after_tpose/removal_estimate.json`。绘图脚本只读取预计算的 `statistics.json`，不再遍历 CSV、读取 NPZ 或重新统计时长。双层环形图外圈为原始数据、内圈为拟保留数据，各圈按自身总时长归一化；图下单独列出 HiPhi 的 T-pose、自碰撞与拟保留占比。默认包含镜像，不应用额外类别 blacklist。BONES-SEED 筛选后保留 129,753 / 142,220 条、263.99 / 288.32 小时；源 CSV 未改写。

BONES-SEED 使用发布方明确的 120 fps，将实际 CSV 路径匹配 `seed_metadata_v004.csv` 的帧数，并抽取五条动作核对实际行数；关键词筛选只影响是否保留整条动作。HiPhi 使用 JSON 中每条文件的 FPS，并抽取五条 NPZ 核对 `fps.npy`。时长按各自 FPS 换算，帧数占比单独记录。

结果输出到 `analysis_output/dataset_statistics/duration_comparison/`：`duration_share.png`、`duration_share.svg`、`statistics.json`、`bones_filter_manifest.json`、`REPORT.md`。预计算入口支持 `--filter-keywords`、`--include-file`、`--hiphi-json` 和 `--collision-json`；绘图入口支持 `--statistics`、`--output`，仅从已有统计绘图。

默认还包含两个尚未下载的 G1 数据时长占位：AMASS-G1 40.67 小时（TextOp III-B 所用、已由论文作者筛选的版本，非全部 AMASS 或某个下载包的实测总量），LAFAN1-G1 2.45 小时（SuperMimic 图 4 的 40 条动作版本）。不再对占位数据做清洗或额外镜像，两圈时长相同；图例以 `*` 标出。来源和统计口径单独保存于 `dataset_statistics/duration_placeholders.json`，后续可替换成正式下载版本的实测结果。`--placeholders` 可指定其他占位 JSON；传入内容为 `[]` 的文件可关闭占位。占位数据不虚构帧数、文件数，全量总帧数为 null，帧数占比仍仅在本地已知帧数的数据内计算。

## 自碰撞扫描

扫描 BONES-SEED CSV 和 HiPhi `motion_actor.npz`：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/self_collision/analyze_self_collision.py --workers 8
```

脚本把每一帧的 root pose 和 29 个关节位置写入 `g1_29dof_with_collision.xml`，先调用 MuJoCo 的 `mj_kinematics` 更新碰撞几何，再调用轻量的 `mj_collision`；不执行完整动力学。世界几何（例如地面）和同一刚体内的几何接触会被过滤。默认逐帧扫描，HiPhi 的 35,252 条索引动作可用 `--stride 2` 做快速初筛；`--executor process`（默认）和 `--executor thread` 均支持。

HiPhi 默认读取 `analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json`，只扫描其中的 `keep_intervals_frames`；输出的碰撞区间仍使用原始 NPZ 帧号。用 `--no-tpose-trim` 可恢复全片扫描。正式两阶段结果使用独立镜像检测：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/self_collision/analyze_self_collision.py \
  --dataset hiphi --workers 12 \
  --output dataset/data_analyze/analysis_output/hiphi_self_collision_after_tpose
python3 dataset/data_analyze/self_collision/estimate_self_collision_removal.py \
  --input dataset/data_analyze/analysis_output/hiphi_self_collision_after_tpose/severe_intervals.json
```

第二步只把命中阈值的动作的 T-pose 保留区间计入拟剔除量，不重复计算已删首尾。两步都不改写原始 NPZ。

当前独立扫描的 35,252 条动作均有效、0 错误；T-pose 删除 14,375,587 帧（79.8644 小时），其余 96,926,174 帧参加自碰撞检测。按 3 cm 连续接触、区间至少 0.5 秒且峰值至少 5 cm 的整条动作剔除规则，拟剔除 2,897 条的保留部分，共 8,860,102 帧（49.2228 小时，占 T-pose 后时长 9.14%）；最终拟保留 88,066,072 帧（489.2560 小时）。详见 `analysis_output/hiphi_self_collision_after_tpose/removal_estimate.json` 和 `analysis_output/dataset_statistics/duration_comparison/`。

HiPhi 镜像可加 `--mirror-policy inherit` 复用原片的碰撞结果，且只读取镜像的帧数、FPS 和文件指纹；缺少原片或时间轴不一致时会独立扫描。输出用 `collision_method=mirror_inherited` 和 `evidence_source` 标明来源。镜像碰撞深度并不严格等于原片：现有独立扫描按 5 cm / 0.5 秒规则有 502 对仅镜像命中、560 对仅原片命中，所以该模式仅适合快速初筛，正式拟剔除统计应保留默认的 `independent`。建议给继承模式指定单独的 `--output` 目录，避免与既有精确结果混用。

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/self_collision/analyze_self_collision.py \
  --dataset hiphi --workers 8 --mirror-policy inherit \
  --output dataset/data_analyze/analysis_output/hiphi_self_collision_inherited
```

结果默认写到 `dataset/data_analyze/analysis_output/self_collision/`：

- `self_collision.json`：总统计和每条源动作的碰撞帧区间、秒区间、接触几何对、源文件 stat 及可选旁车 JSON。
- `statistics.json`：可直接复用的汇总统计。
- `self_collision_checkpoint.jsonl`：逐条动作的中间结果；再次运行时加 `--resume` 可复用源文件未变化的结果。

HiPhi 全量逐帧扫描可单独指定 `--dataset hiphi --output dataset/data_analyze/analysis_output/hiphi_self_collision`。该目录额外包含 `severe_intervals.json`，每段采用零起始、左闭右开的帧区间，并记录秒区间、最大穿透深度和碰撞几何对。严重候选定义为穿透深度至少 3 cm 且连续至少 0.1 秒；这是待复核的无效数据候选，不自动删除源动作。`self_collision.json` 还保留浅层接触记录。

历史未裁 T-pose 的 HiPhi 全量结果还包含 `priority_review.json`：从严重候选中选出区间最大穿透至少 5 cm、区间持续至少 0.5 秒的片段，按最大穿透和持续时间排序。5 cm 是区间最大值，区间内其余帧仍按 3 cm 门槛连续检出；该清单仅用于优先复核，不能直接作为整条动作的删除清单。

用 `python3 dataset/data_analyze/self_collision/estimate_self_collision_removal.py` 计算整条动作的拟剔除量，结果含阈值、汇总和按严重程度排序的逐条清单。默认规则是至少一个连续区间每帧穿透 3 cm、持续 0.5 秒、峰值达 5 cm，则整条动作进入拟剔除集合；可通过 `--peak-depth-m` 和 `--min-duration-s` 调整后复算。历史未裁 T-pose 的结果为 3,486 / 35,252 条、11,894,257 / 111,301,761 帧（约 66.08 小时）；这是估计，源文件尚未删除。样本复核清单位于 `analysis_output/hiphi_self_collision/motion_files.json`。

通用播放器用法见 [visualization/README.md](visualization/README.md)。五条约一分钟的完整原始动作及首尾规则新标记见 [review_endpoints_5](analysis_output/hiphi_tpose/review_endpoints_5/README.md)，通过独立 `motion_files.json` 切换，不遍历整个数据集。此前预测保留在 [review_full_5](analysis_output/hiphi_tpose/review_full_5/README.md)。

## T-pose

查看 20 条复核动作（默认包含前后站立上下文）：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/visualize_hiphi_tpose_viser.py --port 8082
```

当前正式方案按肩角转换边界自适应裁剪：边界后加 0.3 秒，无法可靠定界时该端裁掉 7 秒，保留旧规则更长的删除区间；每条动作首尾均裁剪。镜像沿用原片切点，允许相差 1 帧。全量保留 96,926,174 帧、538.4787 小时，原始 NPZ 不改写。

利用已有测量缓存生成正式 JSON，无需重读 NPZ：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/apply_adaptive_endpoint_trim.py
```

正式结果位于 `analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/`；十条完整原片的新区间标记位于其中的 `review_10/motion_files.json`，也是通用播放器当前默认清单。详见 [T-pose 说明](tpose/README.md)。

以下命令是历史 v2 严格姿态扫描，用于复现基线：

```bash
MPLCONFIGDIR=/tmp/textop-tpose-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/scan_hiphi_tpose.py --detector endpoint --workers 8
```

默认使用 8 个进程，先检测原片；`--mirror-policy inherit`（默认）让同帧率、帧数相差不超过 1 帧的镜像继承原片区间，仅读取镜像帧数头和 FPS。内部切点不变，末尾删除边界对齐镜像实际结尾。当前全部 17,626 对均可继承（含 221 对相差 1 帧），几何检测量由 35,252 降至 17,626 条。`--mirror-policy independent` 可恢复逐条检测。

`scan_checkpoint.jsonl` 支持相同规则和相同源文件下断点续跑；继承的镜像同时校验原片指纹，原片变化也会使镜像缓存失效。规则或脚本变更后需换一个 `--output` 目录，防止混用旧检测结果。可以用 `--executor thread` 切换多线程。先测试速度可加 `--limit 384 --benchmark-only`；继承模式会额外加入抽样镜像依赖的原片。

历史 v2 仅检测首尾校准姿态：确认开头后删除前缀，确认结尾后删除后缀，包含外侧准备/收尾、抬放臂与小动作，中间整体保留。不强求外侧站立；内侧需有双臂放下的证据。每端最多搜索 10 秒且不超过时长 15%，最多删除 12 秒且不超过时长 20%；无足够证据或超出上限时保留并记录原因。它可能漏删首尾，现由自适应规则补齐。

历史基线位于 `analysis_output/hiphi_tpose/full_scan_endpoints_shared_mirror/`；镜像独立检测结果保留在 `full_scan_endpoints/`，历史全时域规则结果保留在 `full_scan/`。正式自适应结果同样使用以下结构，并新增 `endpoint_trim_decisions` 记录各端测量与兜底原因：

- `removal_intervals.json`：每条源文件的 FPS、总帧数、首尾剔除区间、保留区间、检测证据及 `endpoint_review_flags` 未确认原因。
- `statistics.json`、`REPORT.md`：全量、原片、镜像、已有类别 blacklist 过滤后的独立统计，以及按大类汇总。
- `scan_config.json`、`timing.json`：规则、脚本指纹与执行耗时。

仅根据 JSON 重新统计，无需重读原始动作：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/scan_hiphi_tpose.py \
  --from-json dataset/data_analyze/analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json
```

已有完整独立检测 JSON 可用 `--reuse-json <旧 removal_intervals.json>` 转为镜像继承：校验规则、索引和源文件指纹后复用原片结果，包含相差 1 帧的镜像，无需重读动作轨迹。`--from-json` 仅重算统计，保持输入 JSON 的区间策略。

保留帧数按区间补集计算，时长按各文件 `frames / fps` 累加；不会隐式丢弃短片段。今后实际清洗时，应把每个 `keep_intervals_frames` 区间另存成独立动作，不跨越被删间隔直接拼接。
