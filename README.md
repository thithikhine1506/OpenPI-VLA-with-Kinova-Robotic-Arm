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

## Contents

- [Result](#result)
- [Features](#features)
- [System architecture](#system-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Three problems worth reading about](#three-problems-worth-reading-about)
- [What limits it](#what-limits-it)

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

## Features

**Gamepad teleoperation recorder** — Xbox controller drives the arm via
Cartesian twist commands while logging synchronized camera frames and robot
state to HDF5 at a verified 15 Hz. Runtime button calibration handles
controller variation; a live preview window shows exactly what the policy will
see.

**Superset state logging** — every frame stores joint positions, velocities,
torques, end-effector position, orientation as both rotation matrix and
quaternion, commanded and measured gripper, and monotonic timestamps. The
action space is chosen at conversion time, so switching from joint targets to
Cartesian deltas is a converter change rather than a re-collection.

**Coordinate-frame canonicalization** — joint angles are unwrapped within
episodes and canonicalized to a single 2π branch across them, then wrapped
relative to the training mean at inference. Both stages are necessary; skipping
either silently corrupts the state the model sees.

**No-op filtering** — frames where nothing moved are dropped at conversion
(24.5% of raw data here). Expressive single-step policies otherwise learn to
imitate the no-ops and freeze mid-rollout.

**Split deployment** — the policy server runs on a remote GPU while the
real-time control loop stays local to the hardware, connected over websocket.
Measured round trip: 330–430 ms.

**Layered safety** — a held-to-run deadman switch, a joint-speed ceiling, target
rejection for implausible commands, and a table-height floor with automatic
lift-and-recover.

**Evaluation harness** — trial logging by workspace position and checkpoint,
with success rate, grasp rate, and per-region breakdown.

---

## System architecture

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

## Dependencies

### Hardware

| Item | Notes |
|---|---|
| Kinova Gen3, 7-DoF | Firmware 2.6.x, Robotiq 2F-85 gripper |
| Intel RealSense D435i | Fixed third-person mount — **bolt it down** |
| Xbox controller | Wired preferred; wireless sleeps and stops reporting |
| Ethernet to the arm | Gen3 defaults to `192.168.1.10` |
| Training GPU | ≥16 GB (peak measured: 10.69 GiB) |
| Inference GPU | ≥8 GB, or remote |

### Software

Two Python environments are required. Kortex pins `protobuf==3.5.1`, which
breaks on Python 3.10+ (`collections.MutableMapping` was removed in 3.10).
openpi needs 3.11. They cannot share an environment.

**Collection and deployment — Python 3.9**

| Package | Version | Why pinned |
|---|---|---|
| `kortex_api` | 2.6.0.post3 | Match your arm's firmware major version |
| `numpy` | 1.26.4 | `openpi-client` requires `<2.0` |
| `opencv-python` | 4.11.0.86 | 5.x requires numpy ≥2; 4.11 processed the training data |
| `pyrealsense2` | latest | Plain V4L2 returns black frames on the D435i |
| `h5py` | latest | Episode storage |
| `typing_extensions` | latest | Undeclared `openpi-client` dependency |
| `openpi-client` | editable | Websocket policy client |

**Training and conversion — Python 3.11** — managed by `uv sync` in the openpi
repo. Key pins: `jax==0.5.3` (CUDA 12), `lerobot==0.1.0`,
`orbax-checkpoint==0.11.13`, `transformers==4.53.2`.

> LeRobot dataset format matters: `lerobot==0.1.0` reads v2.x. Datasets written
> by v3.0+ are not compatible with openpi's training pipeline.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/thithikhine1506/OpenPI-VLA-with-Kinova-Robotic-Arm.git
cd OpenPI-VLA-with-Kinova-Robotic-Arm
```

### 2. Collection environment (Python 3.9)

```bash
python3.9 -m venv ~/kinova_env
source ~/kinova_env/bin/activate

pip install https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl
pip install -r requirements-collect.txt
```

Verify all three imports before continuing:

```bash
python3 -c "
import kortex_api, cv2, numpy as np, pyrealsense2 as rs
print('cv2', cv2.__version__, '| numpy', np.__version__)"
```

### 3. openpi environment (Python 3.11)

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi
cd ~/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

Skip `--recurse-submodules` — the ALOHA and LIBERO submodules aren't used and
account for most of the clone time.

### 4. Install this project's files into openpi

```bash
cp scripts/kinova_policy.py ~/openpi/src/openpi/policies/
cp scripts/convert_kinova_to_lerobot.py ~/openpi/
```

Then apply the three edits from `scripts/config_snippet.py` to
`~/openpi/src/openpi/training/config.py`: the policy import, the
`LeRobotKinovaDataConfig` class, and the two `TrainConfig` entries.

Verify:

```bash
cd ~/openpi
uv run python -c "
from openpi.training import config
print([n for n in config._CONFIGS_DICT if 'kinova' in n])"
```

### 5. Client library into the collection environment

```bash
source ~/kinova_env/bin/activate
pip install -e ~/openpi/packages/openpi-client
```

---

## Usage

Full walkthrough in **[docs/setup.md](docs/setup.md)**.

```bash
# 1. verify hardware, capture the home pose
python3 scripts/kortex_connection_test.py

# 2. calibrate the controller (once)
python3 scripts/gen3_recorder.py --calibrate

# 3. verify cameras, gripper convention, joystick
python3 scripts/gen3_recorder.py --check --no-wrist

# 4. record demonstrations
python3 scripts/gen3_recorder.py --no-wrist

# 5. convert to LeRobot format
cd ~/openpi
uv run python convert_kinova_to_lerobot.py --dry-run
uv run python convert_kinova_to_lerobot.py

# 6. normalization statistics -- INSPECT THESE before training
uv run scripts/compute_norm_stats.py --config-name pi05_kinova_gen3_lora

# 7. train
uv run scripts/train.py pi05_kinova_gen3_lora --exp-name=run1 --overwrite

# 8. serve
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_kinova_gen3_lora --policy.dir <checkpoint>/5000

# 9. deploy
python3 scripts/gen3_policy_client.py --host <server-url> --dry-run
python3 scripts/gen3_policy_client.py --host <server-url>

# 10. evaluate
python3 scripts/log_trial.py --checkpoint 5000
python3 scripts/log_trial.py --summary
```

---

## Configuration

### Task and hardware — `scripts/gen3_recorder.py`

| Parameter | Default | Notes |
|---|---|---|
| `TASK` | `"pick up the red block and put it in the white bowl"` | Written to every frame; also the inference prompt |
| `ROBOT_IP` | `192.168.1.10` | Gen3 default |
| `HOME_JOINTS_DEG` | 7 angles | From `kortex_connection_test.py`. Every episode starts and ends here |
| `FPS` | `15` | `RefreshFeedback()` costs ~25 ms, capping the loop at 40 Hz before cameras |
| `GRIPPER_INVERT` | `False` | **Verify with `--check`.** Backwards means a policy that opens when it should grip |
| `LIN_SPEED` / `ANG_SPEED` | `0.10` m/s / `25` deg/s | Teleop scaling |
| `GRIPPER_STEP` | `0.04` | Per frame while held; larger reduces no-op frames |

### Conversion — `scripts/convert_kinova_to_lerobot.py`

| Parameter | Default | Notes |
|---|---|---|
| `IMG_SIZE` | `256` | openpi resizes to 224 at train time. Never upscale |
| `JOINT_EPS` / `GRIP_EPS` | `0.002` rad / `0.005` | No-op thresholds |
| `--center-crop` | off | Default squashes 4:3 to square, preserving the full workspace |
| `--filter-noops` | on | Disable only to measure the effect |

### Training — `scripts/config_snippet.py`

| Parameter | Value | Notes |
|---|---|---|
| `action_horizon` | `10` | 0.67 s per chunk at 15 Hz |
| `paligemma_variant` | `gemma_2b_lora` | LoRA rank 16 |
| `action_expert_variant` | `gemma_300m_lora` | LoRA rank 32 |
| `batch_size` | `8` | Fits in ~10.7 GiB |
| `num_train_steps` | `20_000` | ~9 epochs over 17k frames |
| `ema_decay` | `None` | **Must be off for LoRA** |
| `has_wrist_image` | `False` | Single third-person camera |
| `convert_absolute_to_delta` | `True` | Dataset stores absolute joint targets; π₀ trains on deltas. Applying this twice is a silent bug |
| `weight_loader` | `pi05_droid` | DROID-pretrained init. `pi05_base` is the ablation |

### Deployment — `scripts/gen3_policy_client.py`

| Parameter | Default | Notes |
|---|---|---|
| `TRAIN_JOINT_MEAN` | 7 values | **From your own `norm_stats.json`.** Copying these across datasets breaks the state encoding |
| `ACTIONS_PER_QUERY` | `3` | Replan every 0.2 s. At 10 the arm ran open-loop for 0.67 s and drifted |
| `KP` | `2.0` | Position error → joint speed |
| `MAX_JOINT_SPEED` | `20` deg/s | Slow enough to watch and react to |
| `MAX_TARGET_JUMP` | `0.35` rad | Reject and stop on implausible targets |
| `Z_FLOOR` | `0.010` m | Set from the training minimum EE height |
| `Z_ESCAPE` | `0.05` m/s | Upward velocity during floor recovery |
| `GRIPPER_EPS` | `0.02` | Each gripper RPC costs ~25 ms; only send on real change |

> **Deploy at the rate you recorded at.** Mismatched rates change the system
> dynamics and degrade performance even when the policy itself is sound.

Full parameter reference: **[docs/policy.md](docs/policy.md)**

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

## Repository layout

```
scripts/
  kortex_connection_test.py     hardware check, read rate, home pose
  gen3_recorder.py              teleop recording -> HDF5
  convert_kinova_to_lerobot.py  HDF5 -> LeRobot v2.1
  kinova_policy.py              openpi input/output transforms
  config_snippet.py             DataConfig + TrainConfig for openpi
  gen3_policy_client.py         closed-loop deployment
  log_trial.py                  evaluation logging
docs/
  policy.md                     architecture and full parameter reference
  setup.md                      step-by-step reproduction
  evaluation.md                 protocol and results
```

---

## Stack

Python · JAX · [openpi](https://github.com/Physical-Intelligence/openpi) · π₀.₅ ·
LoRA · LeRobot · Kinova Kortex API · Intel RealSense · OpenCV · HDF5

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical
Intelligence. π₀.₅ checkpoints and the DROID-pretrained initialization are
theirs. This repository contains only the Kinova Gen3 adaptation.
