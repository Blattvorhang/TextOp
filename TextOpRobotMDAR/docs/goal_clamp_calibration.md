# Goal clamp 人工标定

运行时限制由 `robotmdar/config/planner_dar.yaml` 中的手工配置决定。训练统计提供观测依据；样本稀少的高度区间需要结合运动先验和闭环结果设限，不应直接用分位数拟合限制曲线。`v(h)` 限制的是 goal 水平位移与有效时间的比值，不保证实际机器人速度。

## 生成诊断统计

在 `TextOpRobotMDAR` 目录运行，使用与目标模型一致的数据、goal encoding、采样窗口和 horizon：

```bash
robotmdar --config-name=refresh_goal_stats \
  data.datadir=/path/to/training_dataset \
  target_dir=./analysis_output/calibration
```

输出统一的 `goal_stats.pkl`，同时保存模型归一化参数和原始量纲的诊断统计。刷新会重算并写入此文件；已有统计包含所需诊断字段时，可以直接绘图，无需重新计算。仅调整 runtime 手工速度参数无需刷新统计。

统计使用 `_compute_goal_stats()` 的采样路径，每批 Torch generator 以批次索引为种子；刷新入口还会用 `cfg.seed` 设置 Python、NumPy 和 Torch 的全局随机种子。可选的 frame-weighted 起点采样使用 Python RNG。复现需要保持数据顺序、batch size、采样权重、goal 构造、数据增强配置和初始随机状态一致，不能只比较 seed 数值。

归一化统计和诊断统计均从原始 goal 构建结果计算，不需要先反归一化。诊断使用的 `goal_clamp_time_min` 和参考高度区间不参与模型归一化尺度计算。

## 查看覆盖程度和人工曲线

```bash
python scripts/plot_goal_clamp.py \
  analysis_output/calibration/goal_stats.pkl \
  --config robotmdar/config/planner_dar.yaml \
  --min-bin-samples 100
```

绘图也兼容缺少新诊断字段的旧 `goal_stats.pkl`，缺失字段会明确提示。输出包括人工速度/半径包络图、按高度的诊断图、CSV 和 `_calibration.yaml`，记录绘图参数与数据元信息。参考高度区间的 pooled 分位数会打印到终端，它只按高度筛选，不能解释为纯行走能力。

- 固定宽度高度桶保留真实占比和空桶；旧等样本量桶只用于兼容读取。
- 空心点表示非空但未达到显示阈值的桶，曲线在这些桶处断开。阈值仅用于提示覆盖不足，不是统计置信界限。
- 计数单位是采样 goal window，相邻窗口可能相关，不能视为独立样本数量。
- 平均速度定义为 `distance / clamp(arrival_time, max(time_min, 1/fps), time_max)`；原始 arrival time 另行保存。
- 混合时间预算的距离分位数仅在诊断图中展示，不直接叠加到固定时间的 radius 曲线上。

## 引入先验并验证

查看多条分位数及各高度区间的样本占比，再结合低姿态运动能力、恢复动作和闭环稳定性选择 `height_low`、`height_full`、`speed_low`、`speed_full`、`time_min`。可以用绘图命令的同名参数临时比较候选曲线，再将验证后的数值写入 planner YAML。

图中的时间 horizon 来自绘图 YAML；实际 planner 从 checkpoint 加载 horizon。比较时应保持二者一致。缺失或 masked time 使用 horizon，过期但有效的时间使用最小 transition duration。

最终验证包括低姿态、起身过程、站立运动、近距离目标和过期目标。保存配置及闭环结果作为参数依据；稀疏区间的统计变化不应自动改变已验证的 runtime 配置。
