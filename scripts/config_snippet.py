"""
openpi configuration for the Kinova Gen3.

These blocks live inside the openpi repo, not standalone:

  1. add to the imports in src/openpi/training/config.py:
         import openpi.policies.kinova_policy as kinova_policy
  2. add LeRobotKinovaDataConfig next to LeRobotLiberoDataConfig
  3. add the TrainConfig entries to the _CONFIGS list

Reproduced here so the repo shows the full configuration without requiring a
diff against upstream openpi.
"""

# ---------------------------------------------------------------------------
# PART 1 -- import
# ---------------------------------------------------------------------------
# import openpi.policies.kinova_policy as kinova_policy


# ---------------------------------------------------------------------------
# PART 2 -- data config
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LeRobotKinovaDataConfig(DataConfigFactory):
    """Data config for Kinova Gen3 (7-DoF) + Robotiq 2F-85.

    Expects a LeRobot dataset written by your converter with top-level keys:
        image, wrist_image, state, actions, task
    """

    # Set False if you collected with only the fixed third-person camera.
    has_wrist_image: bool = True

    # Set False if your converter already emits DELTA actions.
    # We record ABSOLUTE joint position targets, so default is True.
    convert_absolute_to_delta: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack maps YOUR dataset keys -> the keys your robot client will send
        # at inference time. Applied to dataset data only, not at inference.
        repack_map = {
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        # Only map the wrist image if the dataset actually has one -- otherwise
        # RepackTransform raises KeyError before KinovaInputs ever sees the data.
        if self.has_wrist_image:
            repack_map["observation/wrist_image"] = "wrist_image"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_map)]
        )

        # Applied to BOTH dataset data and live inference -- these must match.
        data_transforms = _transforms.Group(
            inputs=[
                kinova_policy.KinovaInputs(
                    model_type=model_config.model_type,
                    has_wrist_image=self.has_wrist_image,
                )
            ],
            outputs=[kinova_policy.KinovaOutputs()],
        )

        # pi0 trains on DELTA actions relative to the first state in each chunk.
        # We record absolute joint position targets, so convert.
        # make_bool_mask(7, -1) -> 7 joints delta, 1 gripper absolute.
        # Grippers are ALWAYS absolute.
        if self.convert_absolute_to_delta:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Tokenizes prompt and action targets. Do not change.
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# ---------------------------------------------------------------------------
# PART 3 -- train configs (add to _CONFIGS)
# ---------------------------------------------------------------------------
    # Kinova Gen3 fine-tuning configs.
    #
    # LoRA fine-tune from the pi0.5 DROID checkpoint. Needs >22.5 GB VRAM.
    TrainConfig(
        name="pi05_kinova_gen3_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotKinovaDataConfig(
            repo_id="thithikhine/kinova_gen3_pick_place",
            base_config=DataConfig(prompt_from_task=True),
            has_wrist_image=False,
            convert_absolute_to_delta=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        num_train_steps=20_000,
        batch_size=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    # Same, but initialized from the generic pi0.5 base. Ablation partner.
    TrainConfig(
        name="pi05_kinova_gen3_lora_from_base",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotKinovaDataConfig(
            repo_id="thithikhine/kinova_gen3_pick_place",
            base_config=DataConfig(prompt_from_task=True),
            has_wrist_image=False,
            convert_absolute_to_delta=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=20_000,
        batch_size=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
