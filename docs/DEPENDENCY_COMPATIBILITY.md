# Dependency compatibility

## Validated workstation

| Component | Validated value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python used by Isaac | 3.12.14 |
| ROS 2 | Humble |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB |
| Driver | 580.178.04 |
| PyTorch / CUDA runtime | 2.10.0+cu128 / 12.8 |
| torchvision | 0.25.0+cu128 |
| Isaac Sim | 6.0.1.0 |
| Isaac Lab | 6.1.14 |
| cuRobo distribution | `nvidia-curobo 0.7.7.post1.dev5` |
| W&B | 0.26.1 |

This is a verified combination, not a claim that every later version works.
Driver 580.178.04 is the validation point; the preflight requires a working
driver and CUDA 12.8-compatible stack rather than guessing an unsupported
minimum.

## Installation layers

1. Install Ubuntu 22.04, NVIDIA driver and ROS 2 Humble using their official
   instructions.
2. Install Isaac Sim 6.0.1 and Isaac Lab 6.1.14 in a dedicated Python 3.12
   environment. Set `GENIESIM_ISAAC_PYTHON` to that interpreter.
3. Install the portable analysis dependencies from `requirements-lock.txt` in
   that environment. CUDA PyTorch wheels may require the PyTorch CUDA index.
4. Install cuRobo in the Isaac environment and verify the reported version.
5. Copy authorized external assets/checkpoints and set `.env` paths.

`environment.yml` is intentionally not a full Isaac environment export. Full
Conda/Kit directories are machine-specific, very large, and not reproducible
through Git.

## Environment variables

See `.env.example`. `PYTHONPATH` is set by wrappers to `<repo>/source`; users
should not hardcode it globally. W&B defaults to offline unless explicitly
changed.

## Capacity

The validated 25-env vector configuration ran on a 16GB RTX 5080. This is an
observed capacity point, not a guaranteed minimum. Start with smoke/1 env and
increase only after finite observation/action/reward and VRAM checks pass.
