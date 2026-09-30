# FainGTwo G2 Stage-1A grasp 재현 가이드

이 문서는 `stage1a-portable-training` 브랜치에서 다른 PC로 G2 Stage-1A를
옮길 때 사용하는 canonical 안내서다. Isaac Sim/Lab, 대용량 asset, dataset,
checkpoint, W&B credential은 Git에 넣지 않는다. 소스는 Git clone으로 받고,
외부 파일은 `faingtwo-stage1a-data-models.tar.gz`로 별도 전달한다.

## 1. Canonical runtime

- Runtime: V3.1 lateral-off, Residual SAC XYZ, HER_FORCE.
- CLOSE authority: deterministic distance/alignment FSM.
- post-CLOSE micro correction: single-contact 0.15 mm, bilateral 0.075 mm.
- frozen Student: high-confidence advisory logging only; FSM을 veto하지 않는다.
- Privileged geometry: teacher/telemetry/validator only. Method E를 제외하면 runtime
  hard gate가 아니며 student privileged input은 0이다.
- Grasp Success: bilateral primary-pad contact가 10 control steps 유지된 Stable만.

Action/timing authority는
`configs/reproducibility/g2_training_authority.json`에 고정되어 있다.

| 항목 | 값 |
|---|---:|
| frame / translation unit | `robot_root` / m |
| final XYZ action bound | 4.5 mm norm, reject-no-clip |
| residual effective authority | 0.45 mm norm |
| teleop normalization divisor | 22.5 mm / normalized unit |
| terminal default / maximum pulse | 1.0 / 4.5 mm |
| control / RGB-D / physics | 50 / 25 / 500 Hz |
| canonical OPEN master target | 0.7853981634 rad |

## 2. Method A–G

| Method | 구성 | Privileged runtime authority | 배포 성격 |
|---|---|---:|---|
| A | V3 CURRENT + HER_FORCE | No | baseline |
| B | V3.1 + historical lateral hard gate | No | ablation |
| C | V3.1 lateral-off + post-CLOSE micro correction | No | best historical candidate |
| D | V3.2 multi-change phase/reward | No | ablation |
| E | Privileged geometry hard CLOSE gate | Yes | diagnostic only |
| F | C + frozen Student advisory | No | deployable advisory |
| G | CURRENT + causal GRU privileged auxiliary | No | privileged target only |

정확한 variant와 authority는 `configs/reproducibility/methods_a_to_g.json` 및
`docs/METHOD_A_TO_G_MATRIX.md`를 따른다. 모든 method의 SAC replay는 현재 live
rollout transition만 사용하며 teacher/offline/keyboard heldout data를 섞지 않는다.

## 3. Safety contract

- runtime hard-stop, forbidden collision, action bound, gripper authority를 유지한다.
- reset은 measured master/passive velocity, aperture OPEN parity 및 fresh geometry
  receipt를 모두 통과한 뒤에만 ACTIVE가 된다.
- source admission은 canonical OPEN 상태의 outer-link4/table clearance를 확인한다.
- direct follower joint, torque/current command는 사용하지 않는다.
- missing/mismatched hash는 fail-closed다.

## 4. 검증 환경

검증 기준점은 Ubuntu 22.04.5, Python 3.12.14(Isaac interpreter), Isaac Sim
6.0.1, Isaac Lab 6.1.14, PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA driver
580.178.04, RTX 5080 16 GB다. `environment.yml`은 CPU/static 분석층만 만들고,
Isaac/PyTorch/CUDA는 NVIDIA/Isaac 설치가 소유한다. ROS2는 Stage-1A simulator
training에 필수가 아니며 Keyboard 전용 repo에는 포함하지 않는다.

## 5. 새 PC 설치

```bash
git clone https://github.com/leeganghyun97/faingtwo.git
cd faingtwo

# Isaac Sim 6.0.1 + Isaac Lab 6.1.14는 별도로 설치한 뒤:
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt

# Google Drive에서 받은 archive를 검증/해제:
sha256sum -c faingtwo-stage1a-data-models-20260930-r12.tar.gz.sha256
tar -xzf faingtwo-stage1a-data-models-20260930-r12.tar.gz
python3 scripts/repro/minimal_training_bundle.py verify \
  --bundle /path/to/faingtwo-stage1a-data-models-20260930-r12/runtime/minimal_bundle
python3 scripts/repro/minimal_training_bundle.py install \
  --bundle /path/to/faingtwo-stage1a-data-models-20260930-r12/runtime/minimal_bundle \
  --frozen-student-checkpoint \
    /path/to/faingtwo-stage1a-data-models-20260930-r12/models/CONDITIONAL_CORAL_FROZEN_STUDENT.pt \
  --write-env --isaac-python /path/to/isaac/python \
  --output-root /path/to/output

# installer가 생성한 ignored .env를 필요에 맞게 검토한다.
${EDITOR:-nano} .env

# Isaac Python에 최소 dependency와 현재 clone의 GenieSim source를 연결:
./scripts/install_minimal_isaac_deps.sh
```

`--write-env`는 기존 `.env`를 덮어쓰지 않는다. 수동 설정을 선호하면 대신
`.env.example`을 `.env`로 복사해 편집한다. `.env`에는 최소
`GENIESIM_ISAAC_PYTHON`, `GENIESIM_ASSET_ROOT`,
`GENIESIM_CANDIDATE_A_USD`, `GENIESIM_PREGRASP_HDF5`, BC/GRU/student/residual
checkpoint 경로를 설정한다. 실제 W&B key는 파일에 넣지 말고 `wandb login`을
사용한다.

## 6. 실행 순서

```bash
./scripts/g2/00_check_env.sh                 # static preflight
./scripts/g2/00_check_env.sh --live A        # asset/hash/package preflight
./scripts/g2/01_static_tests.sh
./scripts/g2/02_app_smoke.sh
./scripts/g2/03_physics_smoke.sh
./scripts/g2/04_open_restore_smoke.sh
./scripts/g2/05_stage1a_smoke.sh A
./scripts/g2/06_train_3k.sh
./scripts/run_method_c.sh --accepted-transitions 6000 --num-envs 10
./scripts/run_method_c.sh --accepted-transitions 6000 --num-envs 10 --execute-live
./scripts/g2/07_train_15k.sh
./scripts/g2/09_eval.sh /path/to/checkpoint_6000.pt 10
```

Method A–G dry-run은 다음처럼 모두 확인한다.

```bash
for method in a b c d e f g; do ./scripts/run_method_${method}.sh --num-envs 10; done
```

현재 canonical vector runtime은 기존 run의 optimizer/replay를 이어 붙이는 resume를
승인하지 않는다. `scripts/g2/08_resume.sh`는 checkpoint를 검사한 뒤 fail-closed로
중단한다. 임의 resume보다 동일 config/hash의 fresh bounded run을 사용한다.

## 7. Training-ready 기준

다음을 모두 통과해야 한다.

1. static preflight와 `01_static_tests.sh` PASS;
2. AppLauncher와 physics first-step smoke PASS;
3. measured OPEN restore, fresh geometry, reset parity PASS;
4. 10-env/100-transition smoke에서 finite observation/action/reward;
5. runtime hard-stop, forbidden collision, action/gripper authority violation, NaN/Inf = 0;
6. external asset/checkpoint SHA-256 일치.

## 8. Dataset / artifact 정책

Git 제외: HDF5/H5, raw RGB-D, `.pt/.pth/.ckpt`, W&B, output, Kit/Omniverse
cache, conda/env, build/install/log 및 ROS2 workspace. 필요한 파일과 SHA-256은
`configs/reproducibility/external_assets.json`과 전달 bundle manifest에 있다.

Keyboard-v3 collection 소스는 별도 ROS2-free repo/export로 제공한다.

```bash
python3 scripts/repro/build_portable_deliverables.py keyboard-source \
  --output "$GENIESIM_DATA_ROOT/releases/g2-keyboard-v3-collection"
```

외부 데이터/BC/checkpoint archive:

```bash
python3 scripts/repro/build_portable_deliverables.py transfer-bundle \
  --output "$GENIESIM_DATA_ROOT/releases/faingtwo-stage1a-data-models"
```

## 9. Known issues

- AppLauncher constructor가 간헐적으로 futex wait할 수 있다. supervisor는 fresh
  top-level process로 최대 3회 bounded retry하며 실패 attempt는 transition 0이다.
- 일부 Isaac native shutdown은 functional report 저장 후 exit 139/SIGSEGV가 날 수
  있다. report/receipt를 먼저 판정하고 이를 학습 PASS로 위조하지 않는다.
- source hash mismatch, missing Candidate-A asset, OPEN parity/table clearance 실패는
  자동 완화하지 않고 fail-closed 처리한다.
- legacy/diagnostic-only 파일은 삭제하지 않는다. canonical entrypoint와 구분해서
  `docs/REPRODUCIBILITY_AUDIT.md`에 남긴다.
