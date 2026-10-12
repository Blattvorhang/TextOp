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
checks. `reanchor_history_representation()` applies one measured tracking
deviation to the selected planner representation; it does not mutate or cache
the controller snapshot.

## Cursor and latency

`reference_motion.cursor_frame` is supplied by the controller and is relative
to that exact reference snapshot. It is the only input used to locate the
reference history window. If latency compensation is enabled, the planner
adds the estimated inference latency as an explicit frame offset and clamps
the result to the available reference window.

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
  latency_compensation: true
  phase_matching:
    enabled: false
```

`generated_history` is no longer a valid configuration concept. The planner
does not maintain a generated motion chain; reset only clears the latest
state, inference guard, and latency estimates for a new episode.

## Naming rules

Use `reference_motion` or `controller_reference_motion` for controller-side
physical arrays. Use `history_repr`, `reference_history_repr`, or
`hybrid_state_history` for planner-side feature tensors. Keep generic
`history_motion` only at training APIs whose public model interface already
uses that name.

