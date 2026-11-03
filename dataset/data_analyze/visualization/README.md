# Motion Viser

通用 G1 动作播放器，从显式文件清单或 NPZ 路径加载；不递归遍历数据目录。
渲染、播放和文件切换逻辑从 T-pose 播放器提取，原 T-pose 入口复用本模块。

## 启动

从仓库根目录运行。默认加载十条完整原始动作及当前正式自适应清洗区间标记：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/visualization/motion_viser.py --port 8084
```

指定独立路径清单：

```bash
/home/hanyi/miniforge3/envs/textop/bin/python \
  dataset/data_analyze/visualization/motion_viser.py \
  --file-list dataset/data_analyze/analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/review_10/motion_files.json \
  --port 8084
```

或使用 `--npz /path/to/one.npz /path/to/two.npz`。浏览器中也可在“文件路径”输入服务器上的路径，点击“打开指定 NPZ”切换；新文件会加入本次会话的下拉列表，清单文件需自行编辑保存。
启动日志打印实际地址；端口被占用时 Viser 会选用后续可用端口。

## 独立路径清单

JSON 中只需提供 `path`，其余字段可选；相对路径以清单文件所在目录为基准：

```json
{
  "version": 1,
  "motions": [
    {
      "path": "motions/example.npz",
      "label": "示例动作",
      "source": "原始文件路径（仅用于显示）",
      "notes": "人工或算法标记说明",
      "annotations": [
        {"label": "待复核区间", "frames": [100, 200]}
      ]
    }
  ]
}
```

也支持 JSON 路径字符串数组，或 TXT 每行一个 NPZ 路径（空行与以 `#` 开头的行忽略）。不会根据 `source` 额外加载文件。
`annotations` 是可选的通用区间标记，不依赖 T-pose 检测器；帧索引从 0 开始，采用左闭右开 `[start, end)`。

## 播放与标记

- 默认播放完整 NPZ；下拉菜单或上一条/下一条切换文件。
- 支持暂停、循环、倍速、逐帧、拖动进度和输入秒数跳转。
- 标记列表显示全部预测区间，提供区间起点/末帧跳转；“当前标记”随播放更新。
- 取消“显示标记”可先自行观察；“未标记”仅表示没有提供标记，不代表没有 T-pose。
- 支持网格/骨架切换和正面、侧面、斜视角。

当前动作格式为项目使用的 G1 NPZ：`fps` 标量、`body_pos_w [T,30,3]`、`body_quat_w [T,30,4]`（wxyz），可选 `joint_pos [T,29]`。
30 个 link 的顺序与原 HiPhi 播放器一致，记录的 body pose 直接驱动模型。`--mjcf` 用于指定兼容的 G1 模型；本工具不推断其他机器人或关节拓扑。

## 本次复核数据

当前十条完整动作及正式自适应删除标记见 [adaptive/review_10](../analysis_output/hiphi_tpose/full_scan_endpoints_adaptive/review_10/README.md)。
历史五条完整动作的首尾规则区间与旧新对照见 [review_endpoints_5](../analysis_output/hiphi_tpose/review_endpoints_5/README.md)。
此前全时域规则的预测保留在 [review_full_5](../analysis_output/hiphi_tpose/review_full_5/README.md)，用于比较漏检和中间误检。
旧的 20 条 T-pose 保持段及上下文仍通过 `tpose/visualize_hiphi_tpose_viser.py --port 8082` 查看。
