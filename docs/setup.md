# Setup and Reproduction

## Hardware

- Kinova Gen3 (7-DoF) with Robotiq 2F-85 gripper, firmware 2.6.x
- Intel RealSense D435i on a fixed mount — **bolt it down and photograph it**
- Xbox controller (wired; wireless works but sleeps)
- Ethernet to the arm; the Gen3 defaults to `192.168.1.10`
- A GPU with ≥16 GB for training (peak measured: 10.69 GiB)

## Two Python environments

Kortex pins `protobuf==3.5.1`, which breaks on Python 3.10+ (`collections.MutableMapping`
was removed). openpi wants 3.11. They cannot share an environment.

**Collection / deployment — Python 3.9:**

```bash
python3.9 -m venv ~/kinova_env
source ~/kinova_env/bin/activate
pip install https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.6.0/kortex_api-2.6.0.post3-py3-none-any.whl
pip install "numpy==1.26.4" "opencv-python==4.11.0.86" h5py pyrealsense2 typing_extensions
pip install -e /path/to/openpi/packages/openpi-client
```

The numpy pin matters: `openpi-client` requires `<2.0`, and opencv 5.x requires
`>=2`. opencv 4.11 satisfies both and is what processed the training data.

**Training / conversion — Python 3.11:**

```bash
git clone -b kinova-gen3 https://github.com/<you>/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
```

Skip `--recurse-submodules` — ALOHA and LIBERO aren't used and account for most
of the clone time.

## Pipeline

### 1. Verify connectivity

```bash
python3 scripts/kortex_connection_test.py
```

Reports actuator count, joint angles, gripper state, achievable read rate, and
a paste-ready home pose. Read-only; it will not move the arm.

**Note the read rate.** `RefreshFeedback()` costs ~25 ms, which caps the loop at
40 Hz before cameras or disk writes. This is why recording runs at 15 Hz.

### 2. Calibrate the controller

```bash
python3 scripts/gen3_recorder.py --calibrate
```

Prompts for each control and writes `padmap.json`. Button numbering varies
between controller models — don't hardcode it.

### 3. Verify everything at once

```bash
python3 scripts/gen3_recorder.py --check --no-wrist
```

Watch the gripper's raw reading with the fingers open, then closed. Getting the
convention backwards trains a policy that opens when it should grip.

### 4. Record

```bash
python3 scripts/gen3_recorder.py --no-wrist
```

`A` starts and saves · `B` discards · `X`/`Y` gripper · `RB`/`LB` wrist roll ·
`8` home · `7` quit.

Aim for ~50 episodes at 20–25 s each. Vary block position across the workspace;
keep the same object. **Watch one episode before recording fifty** — a
misaimed camera produces perfect-looking numbers and an unusable dataset.

### 5. Convert

```bash
uv run python scripts/convert_kinova_to_lerobot.py --dry-run   # inspect filtering
uv run python scripts/convert_kinova_to_lerobot.py
```

### 6. Compute and inspect normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_kinova_gen3_lora
cat assets/.../norm_stats.json | python3 -m json.tool
```

**Read these before training.** Rarely-used dimensions get tiny `std` values,
which blow up after normalization and diverge the loss. A dimension whose
`q01`/`q99` span exceeds 2π is a coordinate-wrapping bug, not high variance.

### 7. Train

```bash
uv run scripts/train.py pi05_kinova_gen3_lora \
    --exp-name=kinova_pick_place --overwrite
```

~4h37m for 20k steps on an A100 at 1.21 it/s. Run 200 steps first to confirm
the loss is finite and falling.

**If checkpointing to a network mount**, verify the files actually landed. A
FUSE mount can accept 10 GB writes, report success, and silently stop syncing —
this cost 8,000 steps of training here.

### 8. Serve

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_kinova_gen3_lora \
    --policy.dir /path/to/checkpoints/.../5000
```

The checkpoint carries its own norm stats, so `compute_norm_stats.py` isn't
needed again at serve time.

### 9. Deploy

```bash
python3 scripts/gen3_policy_client.py --host <server-url>
```

`RB` is a deadman switch — the arm moves only while held. `8` homes, `7` quits.

**Home before every rollout.** Every training episode started from that pose;
starting elsewhere puts the policy at the edge of its distribution before it
begins.

### 10. Evaluate

```bash
python3 scripts/log_trial.py --checkpoint 5000    # second terminal
python3 scripts/log_trial.py --summary
```

---

## Things that will bite you

**Camera alignment.** Compare a live frame against a training frame in
difference mode; move objects until the residual goes dark. A few centimetres
of viewpoint drift translates directly into a grasp offset.

**Never `Ctrl+Z` a script holding the RealSense.** A suspended process keeps
the device and forces a hardware reset:

```bash
python3 -c "
import pyrealsense2 as rs, time
for d in rs.context().devices: d.hardware_reset()
time.sleep(5)"
```

**Copy the frame.** `np.asanyarray(frame.get_data())` returns a view. Retaining
views exhausts the 16-frame pool and every episode caps at exactly 16 frames.

**Match deployment rate to recording rate.** Mismatched rates change the system
dynamics and degrade performance even when the policy itself is sound.
