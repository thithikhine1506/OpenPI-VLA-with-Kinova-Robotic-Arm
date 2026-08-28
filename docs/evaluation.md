# Evaluation

## Protocol

The workspace is divided into a 3×3 grid from the camera's viewpoint:

```
1  2  3     far
4  5  6     mid
7  8  9     near
L  C  R
```

Each trial: home the arm, place the block at a grid position, hold the deadman
until the rollout completes or clearly fails, record the outcome.

| Outcome | Definition |
|---|---|
| success | block ends up in the bowl |
| partial | grasped, then dropped or missed the bowl |
| failure | no grasp achieved |
| void | operator error or infrastructure failure — excluded |

## Results

*(run `python3 scripts/log_trial.py --summary` and paste the output here)*

| Checkpoint | Trials | Success | Grasp (s+p) |
|---|---|---|---|
| 5000 | | | |
| 11000 | | | |

## Per-position breakdown

Failures clustering in a grid cell indicate thin coverage in the training data
for that part of the workspace.

## Qualitative notes

**Checkpoint 11000** reaches the correct region, then grasps air a few inches
from the block. The gripper oscillates (0.4 → 0.8 → 0.3 → 0.9) while the arm
stays still — the signature of a memorized approach with no visual correction.

**Checkpoint 5000** completes the task. In one logged rollout the gripper held
at 0.76 across 135 consecutive queries while j0 traversed −1.22 → +0.19 and j6
traversed 0.34 → 2.43, followed by a clean release over the bowl.

On 54 episodes, the earlier checkpoint generalizes better. Later training
overfits the approach at the expense of visual grounding.
