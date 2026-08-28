# Fine-Tuning π₀.₅ for Pick-and-Place on a Kinova Gen3

A vision-language-action policy trained by imitation on 54 teleoperated
demonstrations, deployed on a 7-DoF Kinova Gen3 arm.

---

## 1. What the policy is

**π₀.₅ (pi05)** from Physical Intelligence's `openpi` framework — a
vision-language-action (VLA) model with three components:

| Component | Role |
|---|---|
| **SigLIP** vision encoder | Encodes 224×224 RGB into patch embeddings |
| **Gemma** language model | Processes the task instruction and fuses it with visual tokens |
| **Action expert** | A flow-matching head that emits continuous action chunks |

The model maps an observation to a sequence of future actions:

```
(image, joint state, "pick up the red block and put it in the white bowl")
        ↓
   10 × 8 action chunk  =  10 timesteps × (7 joint targets + gripper)
```

Flow matching produces smooth continuous actions rather than discretized
tokens, which matters for a manipulation task where trajectory quality
affects grasp success.

**Training method: LoRA.** Base weights stay frozen in bfloat16; only
low-rank adapter matrices train — rank 16 on the PaliGemma backbone, rank 32
on the action expert. This is what makes fine-tuning feasible: peak memory
was 10.69 GiB rather than the ~70 GiB full fine-tuning would need.

**Initialization: `pi05_droid`.** The checkpoint pretrained on the DROID
dataset (76k demonstrations, Franka Panda). The Franka joint-velocity priors
are wrong for a Gen3, but the visuomotor features transfer.

---

## 2. Data

| Property | Value |
|---|---|
| Episodes | 54 (2 failures discarded from 56) |
| Raw frames | 22,987 |
| Frames after no-op filtering | 17,346 (24.5% dropped) |
| Recording rate | 15 Hz |
| Camera | Intel RealSense D435i, fixed third-person, 640×480 → 256×256 |
| Teleoperation | Xbox Series X controller → Cartesian twist commands |
| Task | Single instruction: "pick up the red block and put it in the white bowl" |
| Format | LeRobot Dataset v2.1 |

**State vector (8-dim):** 7 measured joint positions in radians, plus
measured gripper (0.0 open → 1.0 closed).

**Action vector (8-dim):** 7 joint position targets (the *next* frame's
measured position) plus commanded gripper. State uses measured gripper;
action uses commanded — the measured value plateaus at ~0.72 when the block
blocks the fingers, while the command spans the full range.

**Why 15 Hz.** `RefreshFeedback()` on the Kortex API costs ~25 ms. A 30 Hz
loop leaves only 8 ms for camera capture and disk writes, which won't hold.
At 15 Hz there is 41 ms of headroom. Measured `dt` across all episodes:
66.9–68.4 ms against a 66.7 ms target.

**No-op filtering.** Frames where all joints moved < 0.002 rad *and* the
gripper moved < 0.005 were dropped. Expressive single-step policies otherwise
learn to imitate the no-ops and freeze mid-rollout.

---

## 3. Two coordinate problems, and how they were fixed

Both were silent failures — no error, just degraded behavior.

### 3.1 Per-episode 2π branch inconsistency

Kortex reports continuous joints in [0, 360). Joints j0 and j4 sit on the
0/2π boundary at the home pose, so nominally identical start poses landed on
different branches:

```
j0 spread across episode starts: 6.20 rad
j4 spread across episode starts: 6.28 rad
all other joints:               0.00 - 0.14 rad
```

Fix, applied in the converter: wrap the first frame of each episode into
[-π, π] and shift the whole trajectory by that constant offset. Within-episode
continuity is preserved exactly.

Effect on the norm stats:

| Dim | Before (std) | After (std) |
|---|---|---|
| j0 | 1.037 | 0.617 |
| j4 | 0.806 | 0.142 |

The old j4 value was measuring the wrap artifact, not real motion.

### 3.2 Deployment branch mismatch

At inference, wrapping to [-π, π] is not enough. j2 sits on the ±π boundary:
live readings gave **+3.142** while training data centred on **−3.146** — the
same physical pose, 2π apart. Against `q01 = -3.485, q99 = -2.447`, quantile
normalization mapped that to roughly **+11.8** instead of something in [-1, 1].

Fix: wrap the *difference from the training mean*, not from zero.

```python
TRAIN_JOINT_MEAN = np.array([-0.3264, 0.4696, -3.1463,
                             -1.6025, 0.0234, -1.0663, 1.2352])

def canonicalize_state(rad):
    return TRAIN_JOINT_MEAN + wrap_to_pi(rad - TRAIN_JOINT_MEAN)
```

---

## 4. Training parameters

```python
TrainConfig(
    name="pi05_kinova_gen3_lora",
    model=Pi0Config(
        pi05=True,
        action_horizon=10,                        # 10 steps = 0.67 s @ 15 Hz
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",        # LoRA rank 16
        action_expert_variant="gemma_300m_lora",  # LoRA rank 32
    ),
    data=LeRobotKinovaDataConfig(
        repo_id="thithikhine/kinova_gen3_pick_place",
        base_config=DataConfig(prompt_from_task=True),
        has_wrist_image=False,                    # single camera
        convert_absolute_to_delta=True,
    ),
    weight_loader=CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"),
    num_train_steps=20_000,
    batch_size=8,
    ema_decay=None,                               # must be off for LoRA
)
```

**Delta actions.** π₀ trains on actions relative to the first state in each
chunk, but the dataset stores absolute joint positions. `DeltaActions` with
mask `make_bool_mask(7, -1)` converts the 7 joints to deltas and leaves the
gripper absolute — "close to 0.72" is meaningful, "close 0.03 more than
before" is not. `AbsoluteActions` inverts this at inference.

**Normalization.** Quantile-based (`use_quantile_norm=True`), using `q01`/`q99`
computed across all 17,346 frames.

**Training run.** 20,000 steps in 4h37m on an A100-80GB at 1.21 it/s
(~9 epochs). Loss 1.5798 → 0.0048; gradient norm 11.58 → 0.15.

---

## 5. Deployment architecture

The model needs a GPU the lab laptop doesn't have (RTX 4070 Laptop, 8 GB;
π₀.₅ weights alone are 6.2 GiB in bf16). Split across two machines:

```
Lab laptop                          Colab A100-80GB
├── RealSense D435i                 ├── openpi policy server
├── Kortex API → Gen3 (Ethernet)    └── checkpoint (from Drive)
├── Xbox controller (deadman)                  ↑
└── gen3_policy_client.py ──── Cloudflare Tunnel (wss://) ────┘
```

Measured round-trip latency: **330–430 ms**. With a 0.2 s execution window
per chunk this means roughly 60% duty cycle — the arm moves, then pauses
waiting for the next chunk. Motion is visibly discontinuous but functional.

### Control loop

1. Read RealSense frame (640×480 BGR → 256×256 RGB)
2. Read joint state via `RefreshFeedback()`
3. `canonicalize_state()` → 8-dim state vector
4. Send `{image, state, prompt}` over websocket
5. Receive `(10, 8)` action chunk
6. Execute the first N actions at 15 Hz via proportional joint-speed control
7. Repeat

Actions are executed as joint speeds proportional to position error:

```python
err   = wrap_to_pi(target - current)
speed = clip(degrees(KP * err), -MAX_JOINT_SPEED, +MAX_JOINT_SPEED)
```

### Runtime parameters

| Parameter | Value | Reason |
|---|---|---|
| `ACTIONS_PER_QUERY` | 3 | Replan every 0.2 s. At 10 the arm ran open-loop for 0.67 s and drifted off target. |
| `KP` | 2.0 | Position error → joint speed |
| `MAX_JOINT_SPEED` | 20 deg/s | Slow enough to watch and react to |
| `MAX_TARGET_JUMP` | 0.35 rad | Reject and stop on implausible targets |
| `GRIPPER_EPS` | 0.02 | Only send a gripper RPC on real change (each RPC costs ~25 ms) |
| `Z_FLOOR` | 0.010 m | Training minimum EE height was 0.0058 m; below the floor the arm is pressing into the table |
| `Z_ESCAPE` | 0.05 m/s | Upward velocity during floor recovery |

### Safety

- **Deadman switch (RB).** The arm moves only while held; release sends zero
  speeds within one loop iteration (~66 ms).
- **Target rejection.** Any commanded joint more than 0.35 rad from current
  stops the arm and forces a re-query.
- **Table floor.** Below z = 0.010 m the arm stops, lifts at 0.05 m/s, and
  re-queries from a valid pose.
- **Homing.** Button 8 returns to the exact pose every training episode
  started from.

---

## 6. Checkpoint selection

Checkpoints were kept at steps 5000, 10000, and 11000. (Google Drive's FUSE
mount silently stopped syncing after step 12000 during training, so later
checkpoints were lost — the final saved state is step 11000.)

**Step 11000** reached the correct region and then grasped air a few inches
from the block, with the gripper oscillating 0.4 → 0.8 → 0.3 → 0.9 while the
arm stayed still. Signature of a memorized approach.

**Step 5000** produced completed pick-and-place: the gripper held at 0.76
across 135 consecutive queries while j0 traversed −1.22 → +0.19 and j6
traversed 0.34 → 2.43, followed by a clean release.

On 54 episodes, the earlier checkpoint generalizes better. Later training
overfits the approach trajectory at the expense of visual grounding.

---

## 7. Camera alignment

The single largest source of failure was viewpoint drift. The RealSense had
been unplugged and hardware-reset repeatedly between data collection and
deployment. Because the policy has no explicit 3D representation, it infers
the block's position from pixel coordinates learned during training — a small
viewpoint shift translates directly into a spatial offset at the grasp.

Symptom: the arm reaches the right region and closes a few inches off target.

Diagnostic: overlay a live frame on a training frame in difference mode and
move objects until the residual goes dark. After realignment, table contact
stopped entirely (`0 recoveries` across a 264-query run).

---

## 8. Known limitations

- **Latency** dominates the control loop. Overlapping inference with
  execution would remove the ~60% duty cycle penalty.
- **Single task, single object.** All 54 episodes use one red block and one
  white bowl. Generalization to other objects is untested and, on comparable
  work at this scale, unlikely.
- **No wrist camera.** The Gen3's vision module RTSP stream did not connect;
  the policy runs on the third-person view alone.
- **Fragile viewpoint.** The policy depends on the camera staying exactly
  where it was during collection.
- **Checkpoint loss.** Steps 12000–20000 were lost to Drive sync failure.
  Whether they would have performed better than step 5000 is unknown —
  the trend suggests not.
