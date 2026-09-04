# Planner V9: Text Conditioning for Action Style

Last updated: 2026-09-04

This note documents how to reintroduce TextOp-style text conditioning into the
current goal-reaching planner, and how Bones-SEED motion names should be mapped
into useful text/style labels.

The short version:

- The current Bones-SEED packed dataset is structurally compatible with the
  original TextOp text interface: `frame_ann` still uses
  `(start_sec, end_sec, text_label, act_cat)`.
- The original TextOp code feeds `frame_ann[*][2]` into CLIP. It does not feed
  `act_cat` into the denoiser.
- BABEL `proc_label` is a processed free-text action label. It is not an AMASS
  label, and it is not the same thing as the 150-class BABEL category table.
- `dataset/data_process/action_label_2_idx.json` is useful as a standard
  BABEL-compatible action category vocabulary, but it should not be treated as
  the planner's final style prompt vocabulary.
- `_COARSE_RULES` is useful for statistics, sampling, and rough grouping. It is
  too lossy to be the final text supervision for action style.
- Text conditioning is intended to behave like the goal components: it has an
  independent probability mask controlled by `denoiser.cond_text_mask_prob`.
  Setting it to `1.0` makes the text condition always empty while preserving the
  model's fixed token layout.

## 1. Original TextOp Text Pipeline

Original codebase:

```text
/home/lenovo/workspace/planner_control/TextOpRaw/TextOp
```

Relevant files:

```text
dataset/pack_dataset.py
TextOpRobotMDAR/robotmdar/dataloader/data.py
TextOpRobotMDAR/robotmdar/model/clip.py
TextOpRobotMDAR/robotmdar/model/mld_denoiser.py
TextOpRobotMDAR/robotmdar/train/train_dar.py
TextOpRobotMDAR/robotmdar/eval/generate_dar.py
```

The original path is:

```text
BABEL JSON
  -> dataset/pack_dataset.py
  -> train.pkl / val.pkl with frame_ann
  -> SkeletonPrimitiveDataset._load_text_embeddings()
  -> frozen CLIP ViT-B/32 text embedding cache
  -> _extract_single_primitive()
  -> batch condition: text_embedding
  -> train_dar.py y["text_embedding"]
  -> DenoiserTransformer embed_text token
```

### 1.1 BABEL Labels

TextOp uses BABEL annotations aligned to AMASS motions. AMASS provides motion
sequences; BABEL provides English action annotations for those sequences.

The important fields are:

```text
raw_label   annotator-provided text
proc_label  normalized action text, used as the CLIP input
act_cat     one or more action categories, used for statistics/sampling
start_t     annotation start time in seconds
end_t       annotation end time in seconds
```

`proc_label` is the text label consumed by CLIP. `act_cat` is a category list,
not the text condition itself.

In the original packer, each annotation is stored as:

```python
(start_t, end_t, proc_label, act_cat)
```

If no frame-level annotation is available, the original pipeline falls back to a
sequence-level annotation and applies it to the whole motion.

### 1.2 Text Embedding Cache

The original dataloader computes text embeddings by collecting all unique
`frame_ann[*][2]` strings:

```python
for item in raw_data:
    for ann in item["frame_ann"]:
        all_texts.add(ann[2])
```

Then it encodes them with frozen CLIP:

```python
texts = clip.tokenize(raw_text, truncate=True).to(device)
text_embedding = clip_model.encode_text(texts).float()
text_embedding[empty_text, :] = 0
```

The empty string maps to a zero vector, which is the null text condition.

### 1.3 Primitive-Level Text Selection

For every primitive, the original dataloader looks at the future window:

```text
future_start = prim_start + history_len
future_end   = prim_end - 1
```

All annotations overlapping this future window become candidate text labels. If
multiple labels overlap, one is sampled randomly. If none overlap, the primitive
uses the empty string and therefore a zero CLIP embedding.

This is why BABEL frame-level labels are useful for online command switching:
the text can change inside a long sequence.

### 1.4 Denoiser Consumption

The original `DenoiserTransformer` has a dedicated text projection:

```python
self.embed_text = nn.Linear(self.clip_dim, self.h_dim)
```

The transformer sequence is:

```text
timestep token
text token
history tokens
noise token
```

During classifier-free training, the text embedding is randomly replaced with
zero according to `cond_mask_prob`. During inference, `y["uncond"] = True`
forces the text condition to zero for the unconditional branch.

## 2. Current Bones-SEED Packing

Current packer:

```text
dataset/data_process/pack_motion_lib_to_textop.py
```

Current path:

```text
Bones-SEED CSV
  -> convert_soma_csv_to_motion_lib.py
  -> motion_lib PKL
  -> pack_motion_lib_to_textop.py
  -> train.pkl / val.pkl manifest + samples/*.pkl
```

The current packer does:

```python
fine = _extract_action_name(str(name))
coarse = classify_coarse(fine)

"frame_ann": [(0.0, duration, coarse, [coarse])]
```

This is compatible with the original TextOp schema:

```text
(start_sec, end_sec, text_label, action_category_list)
```

However, the semantics differ:

| Aspect | Original TextOp | Current Bones-SEED Pack |
| --- | --- | --- |
| Text source | BABEL `proc_label` | filename -> `_COARSE_RULES` |
| Granularity | frame-level or sequence-level | one sequence-level label |
| Text content | short natural action phrases | coarse category string |
| Category field | BABEL `act_cat` | `[coarse]` |
| Storage | full records in pkl | lazy manifest + `samples/*.pkl` |

The storage difference means the current dataset cannot be used directly with
the unmodified raw TextOp dataloader. The active dataloader in this repo already
supports lazy sample hydration, so this is not a blocker.

## 3. How To Use `action_label_2_idx.json`

The file:

```text
dataset/data_process/action_label_2_idx.json
```

contains the 150 BABEL action categories, for example:

```text
walk
stand
hand movements
turn
interact with/use object
arm movements
step
backwards movement
forward movement
jump
run
stand up
jog
wave
clap
crouch
crawl
sneak
limp
march
zombie
```

This table is valuable as a standard category vocabulary. It should be used to
normalize category-like fields such as:

```text
primary_action
babel_act_cat
sampling_class
evaluation bucket
```

It should not be used blindly as the planner's style prompt vocabulary. Many
BABEL categories describe geometry, phase, objects, or body parts rather than
style:

```text
turn
stop
forward movement
backwards movement
sideways movement
interact with/use object
take/pick something up
place something
touch object
hand movements
arm movements
```

For goal-reaching, those labels either duplicate the goal condition or require
extra grounding that the planner does not currently have.

## 4. Why `_COARSE_RULES` Is Not Enough

The current `_COARSE_RULES` works well as a coarse action distribution tool. It
covers most motion names and produces stable buckets such as:

```text
walk
jog
jump
idle
crouch
gesture
dance
carry
push
fall
injured
```

That is good for:

- dataset summaries
- weighted sampling
- training logs
- broad filtering
- rough first-pass experiments

It is not enough for final text/style supervision because it collapses useful
structure:

| Filename Pattern | Current Coarse Label | Missing Semantics |
| --- | --- | --- |
| `injured_walk_ff` | `injured` | base gait is `walk` |
| `crouch_ff_loop_270_R` | `crouch` | locomotion plus crouched posture |
| `walk_ff_stop_180_R` | `walk` | stop, turn angle, turn side |
| `turn_walk_270` | `walk` or `turn` depending rule order | multi-phase path |
| `jump_over_obstacle` | `jump` | obstacle traversal |
| `walk_the_dog` | `walk` | interaction/object-like context |
| `stand_up_lying_walk` | `fall` or `walk` | recovery then locomotion |

For planner style control, this is the wrong question:

```text
What single class is this whole clip?
```

The better question is:

```text
Which semantic axes are present, and which axes should the planner condition on?
```

## 5. Recommended Bones-SEED Semantic Design

Store sequence-level multi-axis semantics. Do not fabricate frame-level
annotations unless real temporal boundaries are available.

Recommended fields:

```python
semantics = {
    "raw_name": "injured_walk_ff_stop_180_R",
    "motion_text": "walk forward with a limp, stop, and turn right 180 degrees",
    "style_text": "walk with a limp",
    "primary_action": "walk",
    "babel_act_cat": ["walk", "limp", "forward movement", "stop", "turn"],
    "motion_family": "locomotion",
    "gait": "walk",
    "style": ["limp"],
    "direction": "forward",
    "turn_side": "right",
    "turn_angle_deg": 180,
    "phase": ["stop", "turn"],
    "upper_body": [],
    "interaction": [],
    "sequence_type": "multi_phase",
    "goal_compatible": True,
    "style_eligible": False,
}
```

Keep TextOp compatibility:

```python
"frame_ann": [
    (
        0.0,
        duration,
        style_text,
        [primary_action],
    )
]
```

This keeps the original CLIP path simple because `frame_ann[*][2]` remains the
model-facing text. It also keeps existing category-weight code simple because
`frame_ann[*][3]` remains a small sampling category list.

If a future experiment wants full sequence text instead of planner style text,
use:

```python
"frame_ann": [(0.0, duration, motion_text, [primary_action])]
```

Do not put every semantic axis into `act_cat`; that would distort category
statistics and weighted sampling.

## 6. Planner-Facing Text Should Not Duplicate the Goal

The current planner already has explicit goal components:

```text
root position
root orientation / yaw
joint-state target
root velocity
arrival time
scene occupancy
history motion
```

Therefore, planner-facing text should describe how the robot moves, not where it
should go.

Do not include these in `style_text` for goal-reaching:

```text
forward
backward
left
right
clockwise
counterclockwise
turn 90 degrees
turn 180 degrees
stop
start
loop
arc
slowly
quickly
```

Some of these can still be kept in `motion_text` and `semantics`, but they
should not drive the planner text token when the goal branch already specifies
the same quantity.

Examples:

| Raw Name | `motion_text` | `style_text` |
| --- | --- | --- |
| `walk_ff_stop_180_R` | `walk forward, stop, and turn right 180 degrees` | `walk` |
| `jog_arc_cw_loop_003` | `jog in a clockwise arc` | `jog` |
| `injured_walk_ff` | `walk forward with a limp` | `walk with a limp` |
| `crouch_ff_loop_270_R` | `walk in a crouched posture while turning right` | `walk in a crouched posture` |
| `idle_turn_270` | `turn in place` | not style-eligible for reaching |

## 7. First-Stage Style Vocabulary

Use a small, high-confidence prompt set first:

```text
walk
jog
run
walk with a limp
jog with a limp
walk in a crouched posture
jog in a crouched posture
stand
```

`stand` should be used for hold/near-goal phases, not as a reaching style.

Do not enable these in the first stage:

```text
proud
angry
fearful
zombie
heavy-footed
careful
using crutches
wave
clap
salute
carry
pick up
push
jump
hop
```

Reasons:

- some have too few examples in the current packed manifest;
- upper-body actions should become an independent condition later;
- object-like interactions need object or scene state;
- crutch motions are filtered out by the default Bones-SEED filter;
- jumping and hopping are better treated as skills, not locomotion styles.

## 8. Composite Motions

Bones-SEED filenames are sequence-level annotations. They do not provide
frame-level temporal boundaries like BABEL.

Therefore:

- Do not automatically split a clip into multiple `frame_ann` segments unless
  real or verified weak boundaries exist.
- Mark multi-phase clips as `sequence_type = "multi_phase"`.
- Keep their full description in `motion_text`.
- Exclude them from first-stage style supervision if the current primitive may
  see only one phase of the whole sequence.

For example:

```text
faint_stand_up_lying_puke_walk_ff
```

should not be converted into fake frame annotations such as:

```python
[
    (0.0, 2.0, "faint", ["fall"]),
    (2.0, 4.0, "stand up", ["stand up"]),
    (4.0, 6.0, "walk forward", ["walk"]),
]
```

unless those time boundaries are actually known. A safer sequence-level record
is:

```python
semantics = {
    "primary_action": "stand up",
    "babel_act_cat": ["stand up", "walk"],
    "motion_text": "stand up from lying, puke, and walk forward",
    "style_text": None,
    "sequence_type": "multi_phase",
    "style_eligible": False,
}
```

The motion can still be used for VAE or motion-prior training; it just should
not provide clean style supervision for goal-reaching.

## 9. Current Implementation Status

The model-side text path has been restored for the active DAR transformer.

Implemented behavior:

- `robotmdar.model.clip` exposes active CLIP helpers:
  `load_and_freeze_clip()` and `encode_text()`.
- The dataloader can load or compute `<split>_text_embed.pkl` from
  `frame_ann[*][2]`.
- Primitive batches include `text_embedding` with shape `[B, 512]`.
- `train_dar.py::_conditions()` forwards `text_embedding` into the denoiser
  condition dictionary.
- `DenoiserTransformer` has a text token projected by `embed_text`.
- `denoiser.cond_text_mask_prob` controls independent text dropout.
- `force_drop_text` and `y["uncond"]` force the text branch to the null
  condition.
- Planner/eval generation can accept an optional `text_embedding`.
- Older checkpoints missing `embed_text.*` can still be loaded; when no text
  weights exist for inference, text conditioning is disabled for that checkpoint.

The MLP denoiser path is intentionally not part of this restoration.

## 10. Configuration Semantics

Text conditioning has two separate controls:

```yaml
data:
  load_text_embeddings: true
  clip_version: ViT-B/32
  clip_dim: 512

denoiser:
  clip_dim: 512
  cond_text_mask_prob: 0.1
```

`data.load_text_embeddings` controls whether the dataset loads or computes CLIP
embeddings. If it is `false`, the dataloader still emits a `[B, 512]` zero
embedding so the training code path stays stable.

`denoiser.cond_text_mask_prob` controls whether the model can see the text:

| Value | Meaning |
| --- | --- |
| `0.0` | always keep text |
| `0.1` | TextOp/ADAPT-style classifier-free text dropout |
| `1.0` | always drop text, useful for no-text ablations |

Setting `cond_text_mask_prob = 1.0` does not physically remove the text token
from the transformer sequence. It keeps the fixed token layout and zeros the
text content. This is intentionally consistent with goal-component masking.

To avoid the cost of CLIP cache creation in a no-text experiment, use both:

```yaml
data:
  load_text_embeddings: false

denoiser:
  cond_text_mask_prob: 1.0
```

## 11. Recommended Data Update

The next data-layer update should change the packer from:

```python
fine = _extract_action_name(str(name))
coarse = classify_coarse(fine)
frame_ann = [(0.0, duration, coarse, [coarse])]
```

to:

```python
fine = normalize_motion_name(name)
semantics = parse_motion_semantics(fine)
motion_text = render_motion_text(semantics)
style_text = render_style_text(semantics)
primary_action = semantics["primary_action"]

frame_ann = [(0.0, duration, style_text or motion_text, [primary_action])]
```

Recommended parser outputs:

```text
motion_family
gait
style
direction
turn_side
turn_angle_deg
phase
upper_body
interaction
equipment
primary_action
babel_act_cat
sequence_type
goal_compatible
style_eligible
```

Rules should be token/phrase based, not simple substring based. For example,
`crouch_walk` should produce:

```python
{
    "primary_action": "walk",
    "gait": "walk",
    "style": ["crouched"],
    "babel_act_cat": ["walk", "crouch"],
}
```

not:

```python
{
    "primary_action": "crouch"
}
```

## 12. Checkpoint and Experiment Guidance

Old goal-only DAR checkpoints can be used as warm starts after adding
`embed_text`, but they have not learned text semantics. They should not be
expected to respond to text prompts without fine-tuning or retraining.

Recommended experiment order:

1. Train with text embeddings from the existing coarse `frame_ann[*][2]`.
2. Run the no-text ablation with `cond_text_mask_prob=1.0`.
3. Replace coarse labels with `style_text` generated from structured
   Bones-SEED semantics.
4. Compare matched goal settings where only `style_text` changes.
5. Add upper-body text as a separate condition only after locomotion style works.

The key evaluation should fix or match the goal distribution and vary only the
text prompt. Otherwise the model may learn to infer gait from distance and
arrival time rather than using the text condition.
