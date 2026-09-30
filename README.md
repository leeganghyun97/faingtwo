![image.png](./docs/image.png)
<div align="center">
  <a href="https://arxiv.org/abs/2601.02078" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/arXiv-2601.02078-red.svg?logo=arxiv&logoColor=white" alt="arXiv Paper: 2601.02078">
  </a>
  <a href="https://github.com/AgibotTech/genie_sim">
    <img src="https://img.shields.io/badge/GitHub-grey?logo=GitHub" alt="GitHub">
  </a>
  <a href="http://agibot-world.com/genie-sim">
    <img src="https://img.shields.io/badge/Webpage-GenieSim-green?" alt="Webpage">
  </a>
  <a href="https://huggingface.co/datasets/agibot-world/GenieSimAssets">
    <img src="https://img.shields.io/badge/HuggingFace-yellow?logo=HuggingFace" alt="HuggingFace">
  </a>
  <a href="https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset">
    <img src="https://img.shields.io/badge/Dataset-ModelScope-FF6B35?logo=model&logoColor=white" alt="ModelScope Dataset">
  </a>
  <div align="center">
    <a href="https://agibot-world.com/videos/genieSim/modules/heroFullVideoEn.mp4" target="_blank">
      <img src="./docs/videoPlay.png" alt="Play Video" />
    </a>
  </div>
</div>

# Stage-1A A–G reproducibility profile

This branch packages the Candidate-A Stage-1A data/validation/runtime paths
without embedding real data, production assets, credentials, or checkpoints.
The canonical safety default is dry-run. Contact and Bilateral are intermediate
events; **Grasp Success means Stable bilateral contact only**.

## 1. Purpose

The reproducibility profile covers the Keyboard-v3 teleoperation environment,
the exact G2 training URDF/config/action authority, collection and validation,
measured-state OPEN reset gating, source-freeze checks, Method A–G dry-run/live
entrypoints, episode-level grasp evaluation, and bounded safety checks. The
robot/action authority is frozen in
[`configs/reproducibility/g2_training_authority.json`](configs/reproducibility/g2_training_authority.json)
and explained in
[`docs/G2_TELEOP_TRAINING_AUTHORITY.md`](docs/G2_TELEOP_TRAINING_AUTHORITY.md).
The exact method definitions are in
[`docs/METHOD_A_TO_G_MATRIX.md`](docs/METHOD_A_TO_G_MATRIX.md).

## 2. Supported platform

The verified stack is Ubuntu 22.04.5, ROS 2 Humble, Python 3.12.14, NVIDIA
driver 580.178.04, CUDA 12.8 PyTorch 2.10, Isaac Sim 6.0.1, Isaac Lab 6.1.14,
and cuRobo 0.7.7 dev on an RTX 5080 16GB. See
[`docs/DEPENDENCY_COMPATIBILITY.md`](docs/DEPENDENCY_COMPATIBILITY.md).

## 3. Clone

Use the private repository URL supplied by the project owner:

```bash
git clone <private-repository-url>
cd genie_sim
```

Do not substitute the public upstream URL when project-only Stage-1A code is
required.

## 4. External assets

Production USD/mesh packs, Candidate-A overlays, datasets, and learned
checkpoints are intentionally excluded. Obtain them from an authorized source,
verify their SHA-256 values, then configure local paths:

```bash
cp .env.example .env
${EDITOR:-nano} .env
```

See [`docs/EXTERNAL_ASSETS.md`](docs/EXTERNAL_ASSETS.md) and
[`docs/DATA_MIGRATION.md`](docs/DATA_MIGRATION.md). Missing inputs are a
fail-closed `NOT_READY`, not a reason to weaken a hash check.

For transfer between authorized workstations, the smallest supported live
package is the local-only 230 MB Stage-1A bundle. It contains Candidate-A's
complete USD closure, immutable pregrasp evidence, Human-GRU/far-reach BC, and
residual initialization. Isaac Sim/Lab is deliberately not included. Export
and install commands are in
[`docs/EXTERNAL_ASSETS.md`](docs/EXTERNAL_ASSETS.md).

## 5. Environment installation

```bash
./scripts/bootstrap.sh
```

This creates `.venv` for portable analysis/tests. Isaac Sim, Isaac Lab, CUDA,
ROS 2, and cuRobo remain external platform installs. Set
`GENIESIM_ISAAC_PYTHON` to the Isaac interpreter.

## 6. Preflight

```bash
./scripts/preflight.sh --profile static
./scripts/preflight.sh --profile live --method A
./scripts/preflight.sh --profile live --collection
```

Static checks never start Isaac. Live preflight checks the interpreter,
packages, external assets, hashes, the latest G2 training URDF/config, action
scales, cuRobo URDF-mirror parity, and the active reset-fixed runtime
source-freeze profile without issuing robot commands.

## 7. Data collection

Collection is dry-run unless `--execute-live` is present:

```bash
./scripts/collect_data.sh \
  --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3" \
  --episode-id episode-000001

# After inspecting the printed command and passing live preflight:
./scripts/collect_data.sh --execute-live \
  --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3" \
  --episode-id episode-000001 --target-residual-mm 20
```

The terminal collection path controls simulation only; it is not a real-robot
command launcher. Its default key pulse is 1 mm and its metric XYZ norm is
bounded at 4.5 mm. The 22.5 mm value in the teleop environment is the
physical-to-normalized translation divisor, not the Stage-1A per-step motion
authority. Residual SAC has a separate effective authority of 0.45 mm.

## 8. Dataset validation

```bash
./scripts/validate_dataset.sh
./scripts/validate_dataset.sh \
  --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3" \
  --output "$GENIESIM_OUTPUT_ROOT/keyboard_v3_validation.json"
```

The first command validates a tiny synthetic fixture. Dataset frame/unit/rate,
split, outcome, and leakage contracts are documented in
[`docs/DATASET_CONTRACT.md`](docs/DATASET_CONTRACT.md).

## 9. Method A–G

Every wrapper defaults to a 25-env/6K dry-run and calls the existing official
fresh-process supervisor; no method logic is duplicated:

```bash
./scripts/run_method_a.sh
./scripts/run_method_b.sh
./scripts/run_method_c.sh
./scripts/run_method_d.sh
./scripts/run_method_e.sh
./scripts/run_method_f.sh
./scripts/run_method_g.sh
```

Live execution is explicit and refuses an existing output directory:

```bash
./scripts/run_method_c.sh --execute-live \
  --output-root "$GENIESIM_OUTPUT_ROOT/method_c_6k_seed42"
```

Method E is diagnostic privileged authority, not deployable production logic.
Methods F/G require the frozen student checkpoint and preserve their documented
advisory/auxiliary authority boundaries.

The minimum real training route needs neither Methods F/G nor an online W&B
account. After installing the bundle and passing live preflight:

```bash
./scripts/install_minimal_isaac_deps.sh
./scripts/preflight.sh --profile live --method A
./scripts/run_minimal_stage1a_training.sh --wandb-mode offline
```

This runs bounded 10-env/3K V3 CURRENT + HER_FORCE, including fresh-process
startup retry, finite physics smoke, measured OPEN restore, source-freeze,
replay/optimizer updates, and a 3K checkpoint.
The minimum training route uses the frozen far-reach BC and does not invoke
the cuRobo planner. cuRobo remains required for Keyboard/automatic collection,
not for this Stage-1A bootstrap.

## 10. Evaluation and ablation

Reports must include Contact, Bilateral Candidate, Grasp Success (Stable),
Contact→Bilateral, Bilateral→Stable, post-bilateral contact loss, premature
pre-CLOSE contact, and safety violations. Use
`scripts/audit_g2_stage1a_canonical_methods.py` to normalize existing JSON
reports; historical reset-confounded results remain reference-only.

## 11. Tests

```bash
./scripts/smoke_test.sh
python3 -m pytest -q tests/test_reproducibility_release.py
```

The default smoke is CPU/static. A bounded Isaac smoke is opt-in:

```bash
./scripts/smoke_test.sh --isaac G
```

## 12. W&B

Keep credentials outside Git (`wandb login` or a secret manager). Set
`WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_MODE` in local `.env`. Static tests
do not contact W&B; live methods create independent runs.

## 13. Real-robot safety

These wrappers do not authorize hardware. A real-robot entrypoint must require
its own explicit execution flag and `GENIESIM_ALLOW_REAL_ROBOT=1`, then pass
joint/velocity/acceleration, collision, controller, ROS-topic, and emergency
stop preflight. Never bypass the canonical gripper controller with direct
finger, torque, or current commands.

## 14. Common errors

- `LIVE_METHOD_PREFLIGHT_FAILED`: configure Isaac Python and external assets;
  do not alter hashes.
- `OUTPUT_REFUSES_OVERWRITE`: choose a new output directory.
- `SOURCE_FREEZE` failure: restore the exact authorized artifact bundle.
- AppLauncher constructor futex timeout: the fresh-process supervisor records
  a zero-transition failed attempt and retries up to its bounded limit.
- GPU/Isaac unavailable: record `NOT_RUN`; static success is not a live pass.

## 15. Known limitations

Clone alone cannot legally or technically supply proprietary/large external
assets. The working tree contains legacy research scripts with machine-local
absolute paths; canonical wrappers are path-injected, while wholesale legacy
rewrites are intentionally out of scope. Full Method A–G GPU runs and real
robot runtime must be repeated on an authorized workstation after migration.

# 1. Genie Sim 3.0

G2 오른팔 7-DoF keyboard teleoperation과 elbow null-space 제어는
[`docs/G2_7DOF_KEYBOARD_TELEOP.md`](docs/G2_7DOF_KEYBOARD_TELEOP.md)에 정리되어 있습니다.

Genie Sim is the simulation platform from AgiBot. It provides developers with a complete toolchain for environment reconstruction, scene generalization, data collection, and automated evaluation. Its core module, Genie Sim Benchmark is a standardized tool dedicated to establishing the most accurate and authoritative evaluation for embodied intelligence.

The platform integrates 3D reconstruction with visual generation to create a high-fidelity simulation environment. It pioneers LLM-driven technology to generate vast simulation scenes and evaluation configurations in minutes. The evaluation system covers 200+ tasks across 100,000+ scenarios to establish a comprehensive capability profile for models. Genie Sim also opens over 10,000 hours synthetic dataset including real-world robot operation scenarios.

The platform will significantly accelerate model development, reduce reliance on physical hardware, and empower innovation in embodied intelligence. Simulation assets, dataset, and code are fully open source.

# 2. Features
- **High-Fidelity Sim-Ready Assets**: 5,140 validated 3D assets covering five real-world operation fields: retail, industry, catering, home and office. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSimAssets).
- **3DGS-based Reconstruction Pipeline**: Integrate 3DGS-based reconstruction process with visual generative model to synthesize realistic simulation environment with high-precision meshes. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- <b><big>Genie Sim World: A multimodal spatial world model which generates photorealistic 3D world from diverse input types in minutes.</big></b> [GitHub](source/geniesim_world).
- **LLM-Driven Scene Generation**: Natural language-driven generation and generalization which instantly generates diverse simulation scenes through conversational interaction.
- **Large-Scale Synthetic Dataset**: Over 10,000 hours open-source synthetic data across 200+ loco-manipulation tasks with multi-sensor streams, alongside multi-dimensional variations.
- **Synthetic Data Generation**: Efficient toolkit for data collectoin with error-recovery mechanism, supporting both low-latency teleoperation and automated data programming. [ModelScope](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset).
- **Robust and Diverse Benchmark**: Provide 100,000+ simulation scenarios and use LLM to autonomously generate task instructions and evaluation configurations. Discrepancy between simulation and real-world test results is less than 10%.
- **VLM-based Auto-Evaluation System**: Full-spectrum evaluation criteria to provide model's capability profile covering manipulation skills, cognitive comprehension and task complexity.
- **Zero-Shot Sim-to-Real Transfer**: Model trained with our synthetic data exhibits zero-shot sim-to-real transfer capability with superior task success rate compared to model trained with real data.

# 3. Updates
- [4/8/2026] v3.1
  - <b><big>Release Genie Sim World: a multimodal spatial world model for 3D world generation</big></b>
  - Update new benchmarks for instruction following, spatial understanding, manipulation skills, robustness, and sim2real
  - Support human-in-the-loop and distributed reinforcement learning pipeline of RLinf
- [1/7/2026] v3.0
  - Update Isaac Sim to v5.1.0 and support RTX 50series graphic card
  - Provide USD and URDF files of Agibot Genie G2 robot and support whole body control
  - Support 3DGS-based scene reconstruction and convert output to USD format for application in Isaac Sim
  - Release synthetic dataset and corresponding data collection pipeline
  - Add LLM-based features to generate scenarios, task instructions and evaluation configurations
- [7/14/2025] v2.2
  - Provide detailed evaluation metrics for all Agibot World Challenge tasks
  - Add automatic evaluation script to run each task multiple times and record score of all steps
- [6/25/2025] v2.1
  - Add 10 more manipulation tasks for Agibot World Challenge 2025 including all simulation assets
  - Open-source synthetic datasets for 10 manipulation tasks on [Huggingface](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData)
  - Integrate UniVLA policy and support model inference simulation evaluation
  - Update IK solver sdk which supports cross-embodiment IK solving for other robots
  - Optimize communication framework and improve simulation running speed by 2x
  - Update automatic evaluation framework for more complicated long-range tasks

# 4. Documentation

## 4.1 Documentations
Please refer to these links to install Genie Sim and download assets and dataset:

- [User Guide](https://agibot-world.com/sim-evaluation/docs/#/v3)
- [Assets](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)
- [Dataset](https://modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)

## 4.2 Genie Sim Benchmark Leaderboard

<table>
<tr>
<td valign="top">

### GenieSim-Instruction

| Tasks | &pi;<sub>0.5</sub> | GR00T-N1.6 | &pi;<sub>0</sub> |
|:------|:---:|:---:|:---:|
| pick_block_number | **0.73** | 0.28 | 0.17 |
| pick_block_shape | **0.41** | 0.15 | 0.17 |
| pick_common_sense | **0.35** | 0.12 | 0.05 |
| pick_follow_logic_or | **0.58** | 0.56 | 0.26 |
| pick_object_type | **0.81** | 0.56 | 0.27 |
| pick_specific_object | **0.58** | 0.35 | 0.16 |
| straighten_object | **0.66** | 0.33 | 0.46 |
| pick_billiards_color | **0.81** | 0.37 | 0.47 |
| pick_block_color | **0.88** | 0.71 | 0.40 |
| pick_block_size | **0.89** | 0.52 | 0.36 |
| **Avg.** | **0.67** | 0.40 | 0.28 |

</td>
<td valign="top">

### GenieSim-Robust

| Generalization | &pi;<sub>0.5</sub> | GR00T-N1.6 | &pi;<sub>0</sub> |
|:------|:---:|:---:|:---:|
| Reference | **0.92** | 0.58 | 0.46 |
| Instruction | **0.89** | 0.47 | 0.32 |
| Robot Init Base | **0.83** | 0.59 | 0.32 |
| Robot Init Joint | **0.70** | 0.39 | 0.34 |
| Robot End Effector | **0.42** | 0.30 | 0.26 |
| Control Delay | **0.76** | 0.57 | 0.40 |
| Camera Frame Drop | **0.83** | 0.28 | 0.19 |
| Camera Noise | **0.89** | 0.59 | 0.34 |
| Camera Occlusion | **0.93** | 0.59 | 0.41 |
| Camera Extrinsic | **0.39** | 0.27 | 0.22 |
| Ambient Lighting | **0.85** | 0.54 | 0.44 |
| Background | **0.90** | 0.57 | 0.40 |
| **Avg.** | **0.77** | 0.48 | 0.34 |

</td>
</tr>
</table>

### GenieSim-Manipulation

| Tasks | &pi;<sub>0.5</sub> | GR00T-N1.6 | &pi;<sub>0</sub> |
|:------|:---:|:---:|:---:|
| Open Door | **0.60** | 0.35 | 0.46 |
| Hold Pot | **0.35** | 0.00 | 0.14 |
| Pour Workpiece | **0.95** | **0.95** | 0.72 |
| Stock and Straighten Shelf | **0.37** | 0.15 | 0.21 |
| Take Wrong Item Shelf | **0.95** | 0.65 | 0.80 |
| Scoop Popcorn | 0.78 | **0.80** | 0.68 |
| Clean the Desktop | **0.16** | 0.01 | 0.08 |
| Place Block into Box | **0.50** | 0.30 | 0.38 |
| Sorting Packages | **0.45** | 0.14 | 0.13 |
| Sorting Packages Continuous | **0.16** | 0.03 | 0.00 |
| **Avg.** | **0.53** | 0.34 | 0.36 |

### GenieSim-Sim2Real

| Tasks | Sim Env<br>*Sim Data*<br>(sim-to-sim) | Sim Env<br>*Real Data*<br>(real-to-sim) | Real Env<br>*Sim Data*<br>(sim-to-real) | Real Env<br>*Real Data*<br>(real-to-real) |
|:------|:---:|:---:|:---:|:---:|
| Select Color | **0.86** | 0.75 | **0.85** | 0.73 |
| Recognize Size | **0.93** | 0.75 | **0.94** | 0.75 |
| Grasp Targets | **0.72** | 0.54 | **0.71** | 0.58 |
| Organize Items | **0.48** | 0.45 | **0.60** | 0.40 |
| Pack in Supermarket | **0.94** | **1.00** | **0.95** | **0.95** |
| Sort Fruit | **0.90** | **0.90** | **1.00** | **1.00** |
| Place Block into Drawer | **0.80** | **0.90** | **0.85** | **0.90** |
| Bimanual Chip Handover | **0.80** | 0.70 | **0.73** | 0.71 |
| **Avg.** | **0.80** | 0.75 | **0.83** | 0.75 |

> <sup>&dagger;</sup> *Sim Data: 500~1500 episodes of simulation data. Real Data: 500 episodes of real-world data. All models are post-trained from the &pi;<sub>0.5</sub> baseline.*



## 4.3 Support

<img src="./docs/feishu.png" width="30%"/>

## 4.2 Roadmap
- [x] Release more long-horizon benchmark mainuplation tasks
- [x] More scenes and assets for each benchmark task
- [x] Support Agibot World Challenge baseline model
- [x] Scenario layout and manipulation trajectory generalization toolkit
- [x] Provide dockfile and tutorial for scene reconstruction pipeline
- [x] Update motion control toolkit to support Genie G2 teleoperation in simulation
- [x] Support human-in-the-loop and distributed reinforcement learning pipeline of RLinf
- [ ] Upload all assets and dataset on Huggingface
- [ ] Support more tasks and larger models for RLinf

## 4.3 License and Citation

All the data and code within `source/geniesim` and `source/data_collection` are under `Mozilla Public License 2.0`. The `source/scene_reconstruction` project contains code under multiple licenses, for complete and updated licensing details, please see the LICENSE files

Please consider citing our work either way below if it helps your research.

```
@misc{yin2026geniesim30,
  title={Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot},
  author={Chenghao Yin and Da Huang and Di Yang and Jichao Wang and Nanshu Zhao and Chen Xu and Wenjun Sun and Linjie Hou and Zhijun Li and Junhui Wu and Zhaobo Liu and Zhen Xiao and Sheng Zhang and Lei Bao and Rui Feng and Zhenquan Pang and Jiayu Li and Qian Wang and Maoqing Yao},
  year={2026},
  eprint={2601.02078},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.02078},
}
```

## 4.4 References
1. PDDL Parser (2020). Version 1.1. [Source code]. https://github.com/pucrs-automated-planning/pddl-parser.
2. BDDL. Version 1.x.x [Source code]. https://github.com/StanfordVL/bddl
3. CUROBO [Source code]. https://github.com/NVlabs/curobo
4. Isaac Lab [Source code]. https://github.com/isaac-sim/IsaacLab
5. Omni Gibson [Source code]. https://github.com/StanfordVL/OmniGibson
6. The Scene Language [Source code]. https://github.com/zzyunzhi/scene-language
7. COAL [Source code]. https://github.com/coal-library/coal
8. OCTOMAP [Source code]. https://github.com/OctoMap/octomap
9. PINOCCHIO [Source code]. https://github.com/stack-of-tasks/pinocchio
10. URDFDOM [Source code]. https://github.com/ros/urdfdom
11. LIBCCD [Source code]. https://github.com/danfis/libccd
12. LIBMINIZIP [Source code]. https://github.com/switch-st/libminizip
13. LIBODE [Source code]. https://github.com/markmbaum/libode
14. LIBURING [Source code]. https://github.com/axboe/liburing
15. MuJoCo [Source code]. https://github.com/google-deepmind/mujoco
