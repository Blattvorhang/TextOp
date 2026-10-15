# TextOp reference-history planner path

The DAR planner is a stateless inference server in the asynchronous
planner-controller system. It receives the measured state history, the latest
controller reference-motion snapshot, its cursor, and the task conditions on
each inference. It does not retain a generated-plan cache between requests.

## Representation layers

```text
controller_reference_motion
    physical G1 arrays published by the controller

reference_history
    the cursor-selected physical window of history_len + 1 states

reference_history_repr
    the planner motion representation made from that physical window

hybrid_state_history
    the planner representation after history reanchor, sent to DAR
```

Controller reference arrays use IsaacLab joint order and WXYZ root
quaternions. Planner representations use the active dataset feature contract;
conversion includes joint reindexing, quaternion convention conversion, and
feature normalization.

## Inference pipeline

```text
reference_motion.cursor_frame
    -> select history_len + 1 physical reference states
    -> convert controller arrays to planner motion representation
    -> optionally phase-match the selected window
    -> history_reanchor against the measured G1 state
    -> build hybrid_state_history
    -> DAR inference
    -> reconstruct planner motion
    -> publish controller reference motion
```

The selector is implemented by `reference_motion_to_history_repr()`. The
result contains the normalized `history_repr`, the absolute pose needed to
decode it, and the physical root/joint states used for phase matching and seam
checks. `reanchor_reference_motion()` applies one measured tracking deviation
to the raw physical window, then encodes the hybrid history. It does not
mutate or cache the controller snapshot.

## Cursor and latency

`reference_motion.cursor_frame` is supplied by the controller and is relative
to that exact reference snapshot. It is the only input used to locate the
reference history window. The generation endpoint additionally includes
`lookahead_offset_frame` and, when enabled, the estimated inference latency.
V6 selects a window ending at that endpoint, bounded by the available states;
near the beginning it advances the window enough to obtain `history_len + 1`
states. V3 retains its additional departure edge.

With phase matching disabled, the phase offset is zero. The tracking anchor
is the original `cursor_frame`, independently of the generation endpoint.
The selector keeps this cursor state separately because it can lie outside
the selected history window. With phase matching enabled, the matched index
addresses a raw physical window state directly, without a `+1` conversion.

The planner output carries the corresponding integer `new_to_old_ref_offset`. This
is the index in the source reference snapshot of the first returned reference
frame (the seam). For history reanchor it uses the actual selected window end,
including boundary adjustments. For example, cursor 20 with lookahead 5 and
latency 2 gives a V6 seam at source frame 27, while the tracking deviation is
still measured at frame 20. The
controller-side snapshot and planner-side result are separate protocol
objects; neither uses a generic `start_t_ns` field for alignment.

`state.tracking.start_t_ns` is retained as plan metadata and
`state.history_meta.publish_t_ns` remains part of the measured-state history,
but neither timestamp is used to infer the reference cursor.

## History, future, and published motion lengths

The current checkpoint contract is:

```text
history_len = 16 features
future_len  = 64 features
internal reconstruction = 80 physical states
published controller motion = 65 states by default
```

The published motion drops the first 15 history states and keeps the last
history state as the seam, followed by the 64 predicted future states. Set
`pub_all_frames=true` only when the full reconstructed sequence is needed for
inspection.

## Configuration

Reference-history behavior is configured under `reference_history`:

```yaml
reference_history:
  enabled: true
  align_to_g1: history_reanchor
  reanchor_mode: decoupled_world
  reanchor_components:
    hor_pos: full
    vert_pos: full
    orientation: gravity_weighted
    joint: gravity_weighted
  gravity_angle_source: reference  # reference | world
  latency_compensation: true
  phase_matching:
    enabled: false
```

For `decoupled_world`, `gravity_angle_source` selects the two vectors
used for the angle. `reference` (default) compares measured and tracking-anchor
body-frame gravity, `g = R.T @ (0, 0, -1)`. `world` compares measured body-frame
gravity to the fixed vector `(0, 0, -1)`, measuring the actual robot's tilt
from upright independently of reference tilt. Thus matching tilted poses give
zero weight in `reference` mode but nonzero weight in `world` mode.
With `c` the dot product of the two normalized vectors, the initial weight function is
`lambda = clamp(1 - c, 0, 1)`: zero for matching gravity and saturated at one
for discrepancies of 90 degrees or more. Each component can independently
select its reanchor mode via `reanchor_components`: `hor_pos`, `vert_pos`,
`orientation`, and `joint` each accept `open_loop` (weight 0, unchanged reference),
`full` (weight 1, full measured-state correction), or `gravity_weighted`
(weight lambda). The angle source is shared by all gravity-weighted components. Pure world-yaw errors do not
trigger weighted components. This gate measures gravity mismatch, rather than all possible
tracking failures (for example, joints can be obstructed while gravity matches).

The same correction is applied to every physical state in the window:

```text
delta_p = p_real - p_ref_at_tracking_anchor
delta_R = R_real @ R_ref_at_tracking_anchor.T
delta_q = q_real - q_ref_at_tracking_anchor

p_hybrid.xy = p_reference.xy + w_hor * delta_p.xy
p_hybrid.z  = p_reference.z  + w_vert * delta_p.z
R_hybrid    = Exp(w_orientation * Log(delta_R)) @ R_reference
q_hybrid    = clamp_to_joint_limits(q_reference + w_joint * delta_q)
```

Each weight is 0, 1, or lambda according to its component mode. Open-loop joints
retain the original reference without additional clipping. The SO(3) correction
uses shortest-arc interpolation from identity to the measured deviation.
The goal is always expressed relative to the final hybrid history pose, and the
generated future is decoded from that same seam, including when some components
are open loop. Full feedback aligns the tracking anchor to the measured state;
with lookahead, the final seam retains the reference anchor-to-seam displacement.

Set all four entries to `full` for full state feedback, or `open_loop` to retain
the reference window. These component choices apply only to `decoupled_world`;
`se3_left` continues to apply its full rigid correction. Old `gravity_weighting`
configuration is rejected. The legacy
`anchor_state`/`anchor_gate` joint gate is not used by this raw-reference path.
Chunk blending and temporal filtering are not implemented by gravity weighting.

`generated_history` is no longer a valid configuration concept. The planner
does not maintain a generated motion chain; reset only clears the latest
state, inference guard, and latency estimates for a new episode.

## Naming rules

Use `reference_motion` or `controller_reference_motion` for controller-side
physical arrays. Use `history_repr`, `reference_history_repr`, or
`hybrid_state_history` for planner-side feature tensors. Keep generic
`history_motion` only at training APIs whose public model interface already
uses that name.

