# Fine-Tuning a Vision-Language-Action Model on a Kinova Gen3

Fine-tuned **π₀.₅** (Physical Intelligence's VLA) on 54 self-collected
teleoperation episodes and deployed it on a 7-DoF **Kinova Gen3** arm to
perform pick-and-place from a natural-language instruction.

> *"pick up the red block and put it in the white bowl"*

The Gen3 is not in π₀.₅'s pretraining mixture, so this is a full adaptation to
a new embodiment: data collection, coordinate-frame alignment, LoRA
fine-tuning, and closed-loop deployment.

<!-- TODO: add rollout.gif here -->

---

## Result

| Metric | Value |
|---|---|
| Demonstrations collected | 54 episodes, 17,346 frames after filtering |
| Training | LoRA on π₀.₅, 20k steps, 4h37m on an A100 |
| Final loss | 1.5798 → 0.0048 |
| Deployment | Closed-loop at 15 Hz, remote inference over websocket |
| Success rate | *(see [docs/evaluation.md](docs/evaluation.md))* |

**Checkpoint 5000 outperformed checkpoint 11000.** On a 54-episode dataset,
the later checkpoint memorized the approach trajectory: it reached the correct
region and then grasped air a few inches from the block, gripper oscillating
0.4 → 0.8 → 0.3 → 0.9 while the arm stayed still. Checkpoint 5000 completed the
task, holding the gripper at 0.76 across 135 consecutive queries during
transport.

---

## System

```
Lab workstation                          Cloud GPU (A100)
├── RealSense D435i  (third-person)      ├── openpi policy server
├── Kortex API → Gen3 (Ethernet)         └── LoRA checkpoint
├── Xbox controller  (deadman switch)              ▲
└── gen3_policy_client.py ──── websocket over tunnel ─────┘
                                          330–430 ms round trip
```

π₀.₅ weights are 6.2 GiB in bfloat16 — more than the lab laptop's 8 GB card can
hold. Inference runs remotely; the client keeps the real-time control loop
local to the hardware.

---

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Connectivity check | `scripts/kortex_connection_test.py` | Joint state, read rate, home pose |
| Teleop recording | `scripts/gen3_recorder.py` | HDF5 per episode, 15 Hz |
| Format conversion | `scripts/convert_kinova_to_lerobot.py` | LeRobot v2.1 dataset |
| Policy definition | `scripts/kinova_policy.py` | openpi input/output transforms |
| Training config | `scripts/config_snippet.py` | `TrainConfig` + `DataConfig` |
| Deployment | `scripts/gen3_policy_client.py` | Closed-loop rollout |
| Evaluation | `scripts/log_trial.py` | Success rate by workspace position |

Full parameter reference: **[docs/policy.md](docs/policy.md)**
Reproduction steps: **[docs/setup.md](docs/setup.md)**

---

## Three problems worth reading about

Each failed silently — no exception, just degraded behavior.

### 1. RealSense frame-pool exhaustion

Every recorded episode was exactly 16 frames. `np.asanyarray(frame.get_data())`
returns a **view into librealsense's internal buffer**, not a copy. Retaining
those views exhausted the 16-frame pool; every subsequent `wait_for_frames()`
hit its timeout and returned nothing.

```python
return np.asanyarray(color.get_data()).copy()   # .copy() is load-bearing
```

### 2. Per-episode 2π branch inconsistency

Kortex reports continuous joints in [0, 360). Joints j0 and j4 sit on the
0/2π boundary at the home pose, so nominally identical starts landed on
different branches:

```
j0 spread across episode starts: 6.20 rad
j4 spread across episode starts: 6.28 rad
all other joints:               0.00 – 0.14 rad
```

Caught by inspecting normalization statistics: j4's `q01/q99` spanned
0.075 → 6.53, a bimodal distribution masquerading as high variance. Fixing it
dropped j4's std from 0.806 to 0.142 — the old value was measuring the wrap
artifact, not motion.

### 3. Train/serve branch mismatch

At deployment, wrapping to [-π, π] is not enough. j2 sits on the ±π boundary:
live readings gave **+3.142** where training data centred on **−3.146**. Same
physical pose, 2π apart. Against `q01 = −3.485, q99 = −2.447`, quantile
normalization mapped that to roughly **+11.8** instead of something in [−1, 1].

```python
def canonicalize_state(rad):
    """Wrap relative to the TRAINING mean, not to zero."""
    return TRAIN_JOINT_MEAN + wrap_to_pi(rad - TRAIN_JOINT_MEAN)
```

---

## What limits it

- **Camera viewpoint is load-bearing.** The policy infers block position from
  pixel coordinates. Between collection and deployment the RealSense was
  unplugged and reset several times; the resulting shift made the arm reach the
  right region and grasp a few inches off. Realigning it against a training
  frame in difference mode fixed the failure mode entirely.
- **Latency dominates the loop.** 330–430 ms round trip against a 0.2 s
  execution window means roughly 60% duty cycle. Overlapping inference with
  execution would remove this.
- **One object, one bowl.** All 54 episodes use the same red block. Comparable
  published work at this scale fails on unseen objects; no reason to expect
  otherwise here.
- **No wrist camera.** The Gen3's vision module RTSP stream would not connect;
  the policy runs on the third-person view alone.

---

## Stack

Python · JAX · [openpi](https://github.com/Physical-Intelligence/openpi) · π₀.₅ ·
LoRA · LeRobot · Kinova Kortex API · Intel RealSense · OpenCV · HDF5

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical
Intelligence. π₀.₅ checkpoints and the DROID-pretrained initialization are
theirs. This repository contains only the Kinova Gen3 adaptation.
