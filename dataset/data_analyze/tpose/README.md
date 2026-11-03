# HiPhi T-pose 清洗

## 运行入口

当前清洗只需运行 `apply_adaptive_endpoint_trim.py`。它读取已有的首尾测量缓存与历史姿态基线 JSON，生成正式剔除/保留区间和统计，不重新读取原始 NPZ：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/apply_adaptive_endpoint_trim.py
```

正式输出是 [`full_scan_endpoints_adaptive/removal_intervals.json`](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json)。按 `motions[*].keep_intervals_frames` 裁剪训练片段；`endpoint_trim_decisions` 记录每端自适应/兜底的依据。原始 NPZ 不修改。

| 文件 | 角色 |
| --- | --- |
| `apply_adaptive_endpoint_trim.py` | **正式清洗入口**：读取测量缓存与基线 JSON，处理原片及镜像，输出清洗 JSON |
| `analyze_endpoint_trim_duration.py` | 测量数据发生变化时才运行：从原片计算首尾肩角转换与时长分布 |
| `scan_hiphi_tpose.py` | 历史基线扫描器；正式入口还复用其中的区间、镜像继承及统计函数 |
| `detect_endpoint_tpose.py`、`analyze_hiphi_tpose.py` | 历史严格姿态检测及特征提取，供基线扫描器复现 |
| `visualize_hiphi_tpose_viser.py` | 查看早期 20 条保持段及上下文；完整动作和正式标记使用通用播放器 |

历史抽样和五条对照脚本已清理；已导出的复核 NPZ、对照图和清单保留在 `analysis_output/hiphi_tpose/`。

通用的机器人渲染和播放逻辑已提取到 [`../visualization/motion_viser.py`](../visualization/motion_viser.py)，通过 JSON/TXT 路径清单或 `--npz` 指定文件。
本目录的 `visualize_hiphi_tpose_viser.py` 保留 T-pose 样例的保持段、站立上下文和完整原片视图，复用通用播放器。

## 首尾裁剪时长：实际统计结论

进一步人工复核只删一端/两端未删的十条完整原片后，确认除 `Scouring-scour_0025` 开头不明显外，其余首尾都有 T-pose。用户要求所有原片与镜像都考虑裁剪首尾，允许损失部分有效动作；2 秒仅为举例，不能直接成为清洗参数。

```bash
MPLCONFIGDIR=/tmp/textop-tpose-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/analyze_endpoint_trim_duration.py --workers 8
```

8 进程读取全部 17,626 条独立原片，约 105 秒；镜像不重复测量。首尾转换边界中位数分别 3.40 / 3.32 秒，P95 为 6.64 / 6.32 秒，P99 为 8.38 / 8.58 秒。2 秒只能覆盖约 13% / 11% 的可测边界，明显不够。

已采用首尾转换点自适应裁剪并加 0.3 秒余量；无法定界时该端使用 7 秒兜底，保留旧规则更长的删除区间。含镜像正式保留 96,926,174 帧、538.4787 小时，删除 79.8644 小时（12.916%）；35,252 条全部首尾均裁剪，无删空或不足 64 帧的剩余动作。17,626 个镜像全部沿用原片切点并对齐实际末帧。

独立原片共 35,252 个端点，其中 33,795 个自适应、1,457 个兜底。测量预算 min(12 秒, 20% 时长) 用于判断测量是否可靠，不再阻止兜底、额外余量或旧规则较长区间的保留。各端决策与测量状态保存在 `endpoint_trim_decisions`，旧 `detected_cycles` / `endpoint_review_flags` 仅作历史证据。

正式结果：[`full_scan_endpoints_adaptive/removal_intervals.json`](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/removal_intervals.json)、[统计报告](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/REPORT.md)、[十条原片复核清单](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/review_10/README.md)。时长占比统计与通用播放器默认入口均已切换。

若要纯固定规则，每端 7 秒覆盖约 96% / 97% 的可测边界，但只保留 481.25 小时。覆盖率来自自动肩角测量，不等于人工标注 recall；未覆盖和不可测端点不能认定为没有 T-pose。

完整证据、方法和方案对比见 [`endpoint_duration_analysis/REPORT.md`](../analysis_output/hiphi_tpose/endpoint_duration_analysis/REPORT.md)，含十条人工复核样本的肩角曲线、全量逐条测量缓存及累计分布图。原始 NPZ 与历史 JSON 均保留。

此前 `full_scan_endpoint_min2s/` 仅为误把示例参数当作规则的历史试算，不能作为正式清洗结果；对应的试算脚本已删除。

## 历史基线：首尾校准规则 v2

人工完整复核五条原片后，确认每条首尾都有校准 T-pose，中间 `drop_0093` 的重复抬放臂属于真实动作。因此默认改为 `scan_hiphi_tpose.py --detector endpoint`，中间不做 T-pose 删除。

默认 `--mirror-policy inherit`：先检测原片，同帧率且帧数相差不超过 1 帧的镜像继承原片区间。内部切点不变，末尾删除边界对齐镜像实际结尾。镜像无需读取全轨迹，只检查数组头与 FPS；不匹配或缺少有效原片结果时独立检测。当前全部 17,626 对均可继承（含 221 对相差 1 帧），几何检测量减少 50%。`interval_source`、`evidence_source` 记录原片路径，继承记录中的 `detected_cycles` 保留原片证据，实际切片使用 `remove/keep_intervals_frames`；镜像自己的 source/stat/blacklist 信息保留。缓存同时跟踪原片和镜像的文件指纹。

- 每端最多搜索 10 秒且不超过时长 15%，每端最多删除 12 秒且不超过时长 20%；开头只选最早、结尾只选最晚的合格周期。
- 连续至少 0.6 秒满足：肩外展 65°–120°、左右差 ≤20°、上臂侧向余弦 ≥0.90、肘偏折 ≤25°、躯干倾斜 ≤25°、root 平面速度 ≤0.12 m/s，且腿部直立、双脚接近地面；0.4 秒内肩角极差 ≤10°。完整阈值见 `ENDPOINT_RULES`。
- 不再要求记录外侧也有站立：可以第一帧就在 T-pose，或最后仍停留在 T-pose。
- 内侧 2 秒内须有至少 0.25 秒双臂放下、躯干直立的姿态；允许腿部仍在行走过渡，不再比较前后 root/腿部姿态一致性。
- 开头删除 `[0, 内侧低臂窗口末尾)`，结尾删除 `[内侧低臂窗口起点, 总帧数)`。准备、抬放臂和外侧收尾小动作一并删除，中间只剩一条连续动作。
- 证据不足或超过上限时保留，记录 `endpoint_review_flags`。五条样例的人工反馈不等于全数据集真值，不无条件固定裁掉每条首尾。

历史 `PRECISION_RULES` 与 20 条保持段仍保留。`--detector cycle` 可运行原来的全时域“站立—T-pose—站立”逻辑；使用新的输出目录避免混用旧指纹缓存。该逻辑既漏掉不完整首尾，也可能误检中间真实抬放臂动作，已不作为默认清洗策略。

## 输出

所有结果继续放在 [`../analysis_output/hiphi_tpose/`](../analysis_output/hiphi_tpose/)，避免改变已有 NPZ 的路径。

- [`review_20/README.md`](../analysis_output/hiphi_tpose/review_20/README.md)：20 条已人工检查的复核样例。
- [`review_full_5/README.md`](../analysis_output/hiphi_tpose/review_full_5/README.md)：五条完整原始动作（含命中/未命中），用于检查 precision/recall；独立路径清单为 `review_full_5/motion_files.json`。
- [`review_endpoints_5/README.md`](../analysis_output/hiphi_tpose/review_endpoints_5/README.md)：同五条原片的历史首尾标记，包含旧新曲线对照。
- [`VIEWER.md`](../analysis_output/hiphi_tpose/VIEWER.md)：Viser 启动和使用方式。
- [`full_scan_endpoints_adaptive/REPORT.md`](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/REPORT.md)：当前正式自适应清洗统计。
- [`full_scan_endpoints_shared_mirror/REPORT.md`](../analysis_output/hiphi_tpose/full_scan_endpoints_shared_mirror/REPORT.md)：历史严格姿态基线、镜像继承区间的全量统计。
- [`full_scan_endpoints_shared_mirror/removal_intervals.json`](../analysis_output/hiphi_tpose/full_scan_endpoints_shared_mirror/removal_intervals.json)：历史基线的首尾剔除/保留区间，带原片来源。
- [`full_scan_endpoints/REPORT.md`](../analysis_output/hiphi_tpose/full_scan_endpoints/REPORT.md)：此前首尾规则、镜像独立检测的统计，保留用于比较。
- [`full_scan/REPORT.md`](../analysis_output/hiphi_tpose/full_scan/REPORT.md)：历史全时域周期规则下的全量统计，供比较。

命令、并行和断点续跑说明见上级 [README](../README.md)。源文件保持只读，后续清洗应将每个保留连续区间单独存储。

## 历史结果：原片检测，镜像继承

由已有完整 JSON 转换并校验所有源文件指纹，未重读动作轨迹，也未重新运行姿态检测。
全部 17,626 条镜像继承原片结果，其中 221 条相差 1 帧，统一沿用原片切点并对齐尾部删除边界。
全量建议剔除 9,998,775 帧（55.5488 小时，8.983%），保留 101,302,986 帧、562.7944 小时。
原片结果不变：保留 50,651,492 帧、281.3972 小时。已有 blacklist 过滤后（含镜像）保留 96,749,016 帧、537.4945 小时。
全量已逐条验证区间继承、镜像自身文件信息、证据来源、帧数守恒和统计一致性。

历史扫描器默认按上述基线规则处理。已有完整独立检测 JSON 可这样转换，旧结果保留：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/tpose/scan_hiphi_tpose.py \
  --reuse-json dataset/data_analyze/analysis_output/hiphi_tpose/full_scan_endpoints/removal_intervals.json
```

## 首尾规则全量结果（v2，镜像独立检测）

16 进程完成 35,252 条索引动作扫描，用时约 4 分 2 秒（另复用此前 512 条基准缓存），读取错误 0 条。
共 28,611 条文件有建议删除；确认开头 24,581 条、结尾 24,593 条，两端均确认 20,563 条。
其余 21,330 个端点未确认，保持原始数据并在 JSON 中记录原因；这些命中数量不等于人工标注的 precision/recall。

| 口径 | 原始小时 | 剔除帧数 | 剩余帧数 | 剩余小时 |
| --- | ---: | ---: | ---: | ---: |
| 全部（含镜像） | 618.3431 | 9,943,401 | 101,358,360 | 563.1020 |
| 原片（不含镜像） | 309.1714 | 4,999,359 | 50,651,492 | 281.3972 |
| 已有 blacklist 过滤后（含镜像） | 590.8587 | 9,555,995 | 96,798,565 | 537.7698 |

全量建议删除 55.2411 小时，占原始时长 8.934%。逐条验证了只删除前缀/后缀、各文件保留一个连续中间区间、帧数守恒和每端删除上限；仅从 JSON 独立重算统计，与全量结果完全一致。五条复核动作的完整副本未改写，浏览器已验证新首尾标记与跳转；`drop_0093` 的两段历史中间误检均保留。

## 历史全时域规则结果（v1）

2026-10-22 扫描 `filenames.txt` 全部 35,252 条动作（含镜像），16 个进程用时约 4 分 13 秒，读取错误 0 条。
共命中 567 条文件、599 个周期，建议剔除 119,840 帧（39 分 56.8 秒，约占 0.108%）。
剔除后保留 111,181,921 帧、617.6773 小时；不计镜像则保留 55,590,889 帧、308.8383 小时。

当前结果仅覆盖满足完整“稳定站立—T-pose—稳定站立”条件的周期。严格规则的剔除占比较低，不能据此认为数据中的大部分 T-pose 已被覆盖；本次保持人工认可的规则，没有为增加删除量而放宽条件。
已逐条验证所有文件的剔除与保留区间互补、帧数守恒，并从 JSON 独立重算统计，与全量扫描结果一致。
