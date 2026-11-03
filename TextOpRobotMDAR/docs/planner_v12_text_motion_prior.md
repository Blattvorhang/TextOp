# Planner v12: Text Motion Prior

## 目标

TextOp 的文本条件直接忠于 BONES-SEED 原始标注，不再从长描述中用正则表达式重构 `sequence_core`、`event_core` 或人工 ontology。文本只做大小写和空白规范化。

## 三个语言层级

每个 motion 保存三种粒度的标注：

1. **core**：原始 `content_type_of_movement`，覆盖整个 sequence；保留完整字符串，不拆分逗号标签、不做 synonym 合并。
2. **sequence**：原始 `content_short_description`，覆盖整个 sequence。
3. **segment**：temporal JSONL 中与时间窗口对应的原始 event `description`，保留其起止时间。

训练 primitive 的文本在 data loading 时随机抽取，默认概率为 `core=0.4, sequence=0.3, segment=0.3`。如果某一级缺失，其概率自动归一化到当前可用级别；没有任何文本时返回空字符串。固定概率贯穿整个训练过程，不使用 curriculum。

## 配置

数据目录可放置 `text_sampling.yaml`，也可在 dataset 配置中通过 `text_sampling_config` 指定路径：

```yaml
core: 0.4
sequence: 0.3
segment: 0.3
```

三项必须是有限的非负数，且总和大于零。未提供配置时使用上述默认值。旧的四字段 `frame_ann` 数据仍可被 dataloader 读取。

## `frame_ann` 字段定义

v12 的 `frame_ann` 是一个按层级组织的字典：

```python
frame_ann = {
    "core": [{"text_labels": ["walking, turning"],
              "act_categories": ["walk"]}],
    "sequence": [{"text_labels": ["a person walks", "a person turns"],
                   "act_categories": ["walk"]}],
    "segment": [{"start_time": 1.2, "end_time": 2.0,
                  "text_labels": ["the person gradually turns left"],
                  "act_categories": ["walk"]}],
}
```

每个层级的 `text_labels` 支持多个字符串；先按 `core/sequence/segment` 概率选层级，再在该层级的文本中等概率随机选择。`core` 和 `sequence` 通常覆盖整个 motion；`segment` 使用 temporal JSONL 的 event 起止时间。`act_categories` 只用于加权统计，不参与 `is_recovery` 判断。

## 数据流

`pack_motion_lib_to_textop.py` 只负责关联原始 metadata 和 temporal JSONL、对文本做轻量规范化、保存时间范围，不执行动作分类或描述重写。`SkeletonPrimitiveDataset` 在每个 primitive 的 future overlap 范围内收集可用标注，再依据配置概率选择一个层级和对应文本。CLIP embedding cache 收集三层中出现的全部字符串，因此采样概率变化无需重新打包，只需重建或补齐 embedding cache。

该设计表达同一 motion 的 categorical → short natural language → temporally detailed language augmentation，同时保留旧 `frame_ann` 接口和 coarse action category 的统计路径。

## 面向 humanoid planner 的重新审视

结合 planner、motion generator 和 WBC 的职责划分，文本主要应该表达 **motion intent / semantic prior**，而不是几何目标或低层执行约束。`walk to the chair` 中的文本只适合告诉 motion generator 选择哪一类运动；目标位置、速度、朝向、接触和关节参考应由 geometry/state condition 提供，WBC 不应直接理解文本。

因此，最小且最稳定的主文本可以直接定义为：

```python
task_text = normalize_text(content_type_of_movement)
```

推荐字段优先级：

```text
content_type_of_movement  -> 主文本（motion intent）
content_uniform_style     -> 可选 style prior，不作为默认主文本
content_short_description -> 辅助自然语言监督
temporal segment text     -> 细粒度描述实验，不作为默认 planner condition
content_body_position     -> pose/state 属性，不作为动作任务文本
```

主文本提取不需要正则、关键词 ontology、拆分或 synonym merge，只做 lowercase 和 whitespace normalization；例如 `walking, turning` 必须作为一个完整原始标签保留。缺失时才回退到 `content_short_description`，再缺失则使用空文本并记录缺失率。

对于默认训练配置，建议将 `core` 视为主条件，`sequence`/`segment` 仅作为 auxiliary text augmentation。若实验目标是验证几何条件下的 semantic prior，建议先使用 `core=1.0, sequence=0.0, segment=0.0` 建立基线，再单独做自然语言增强 ablation，而不是把三者混合后的结果当作唯一主模型。
