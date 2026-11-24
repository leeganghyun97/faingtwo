# 원본 증거와 추가 자료 요청 지도

이 문서는 큰 파일을 포함하지 않는다. 새 컴퓨터에서 어떤 자료가 필요한지와, 원래 작업공간에서 찾을 상대 경로를 연결한다. 경로는 기준 스냅샷 `x2_locomotion_migration_20261124`의 repository root 기준이다.

## 우선 읽을 소형 문서

| 목적 | 원본 상대 경로 | 왜 필요한가 |
|---|---|---|
| 전체 방법론/계보 | `docs/methods/METHODS_KO.md` | 800개 최상위 실험 분류, source system 분리, contracts |
| 학습·재개 계약 | `docs/methods/TRAINING_AND_RESUME_KO.md` | strict resume/warm start, PPO/RNN 경계 |
| 계열 색인 | `docs/methods/EXPERIMENT_FAMILY_INDEX_KO.md` | 실험 family와 관련 artifact 위치 |
| 결과와 한계 | `docs/results/RESULTS_AND_LIMITATIONS_KO.md` | success claim 방지, known missing artifacts |
| 최신 V18/R33/R34 검토 | `docs/results/LATEST_R33_R34_REVIEW.json` | bounded exposure 범위와 `NOT_COMPARABLE` 비교 이유 |
| 환경 | `docs/runtime/ENVIRONMENT_KO.md` | Conda/Isaac/runtime profile |
| porting | `docs/runtime/PORTING_KO.md` | source root, asset, symlink, new host risks |
| 자산 색인 | `docs/runtime/ASSET_CATALOG.json` | URDF/USD/mesh 외부 의존성 |
| 새 컴퓨터 복원 | `docs/NEW_COMPUTER_SETUP_KO.md` | 복원 순서와 storage caveat |

## 중요한 개별 실험/결과 위치

| 주제 | 기준 run/folder | 사용할 때의 주의 |
|---|---|---|
| U5 lateral adapter/ankle contact | `x2_u5_lateral_landing_diagnostic_20261105/` | small adapter output과 native contact causal claim을 분리 |
| U100 landing impact | `x2_u100_landing_dynamics_20261106/` | native contact buffer authority가 선행 조건 |
| U230 causal/landing | `x2_u230_*_20261109` 및 `x2_u230_u1230_*_20261110/` | target/q/qd와 impact pre/post timebase를 확인 |
| U730 user pose/upper body | `x2_u730_*_20261111/` | Viser motion plausibility와 physical gait를 분리 |
| lift acquisition | `x2_lift_acquisition_*`, `x2_u20_u40_lift_recovery_audit_20261114/` | mm/micro-lift metric의 gate 정의를 함께 읽기 |
| task-space touchdown anti-hack | `x2_taskspace_touchdown_antihack_10mm_20261116/` | touchdown bonus/reward hacking audit |
| Teacher weight transfer | `x2_teacher_weight_transfer_recovery_20261120/` | Support Ready/Swing Unloaded/Transfer Ready를 구분 |
| 90-control APA/pre-unload | `x2_weight_transfer_90clock_*_20261120/` | phase clock, stance share, swing Fz, COM alignment |
| V9–V18 policy prior/support first | `x2_v9_*` … `x2_v18_support_first_method_redesign_20261123/` | latest F0/F1/F2 frozen contracts and receipts |

## strict reproduction에 필요한 최소 입력

새 호스트에서 old run을 그대로 재생하려면 다음 묶음이 같은 lineage/SHA로 있어야 한다. 하나라도 빠졌다면 strict reproduction이 아니라 reconstruction이라고 기록한다.

1. **소스**: exact commit/working tree와 Python import order. HDMI, optional external `RoboJuDo-Plus` sibling은 별도다.
2. **runtime**: compatible Isaac Sim/Lab build, Python executable, GPU driver/CUDA/PhysX compatibility, Conda/pip locks.
3. **asset**: robot registration source, URDF/USD/mesh/collision variant, material/friction/mass configuration, joint names/order/limits.
4. **policy state**: checkpoint, optimizer state, observation normalization, reward/phase curriculum progress, RNG state, recurrent hidden-state/resume contract where applicable.
5. **environment contract**: control dt, physics dt/decimation, seed/spawn/reset, number of envs, commands, terrain, guard/native safety limits.
6. **evaluation trace**: immutable actor checkpoint SHA, applied action/target, q/qd, contact/body/pair, reset/terminal ledger, timebase metadata.

## 이관본의 알려진 미완전성

- 역사적 checkpoint 약 3,448개(약 18.06 GB)는 삭제되어 복구 불가로 기록됐다.
- `/data` 아래의 역사적 외부 의존성은 완전한 portable bundle이 아니다.
- 원본 symlink 중 일부와 URDF/MJCF references는 새 호스트에서 재해결이 필요하다.
- 저장된 artifacts 수·파일명은 gait success의 증거가 아니다.

## 추가로 작은 파일만 전달할 수 있을 때의 요청 우선순위

원본 전체가 너무 크면 아래를 작은 단위로 요청한다.

1. chosen baseline checkpoint 하나와 SHA-256, normalization/progress/RNG sidecar
2. 해당 run의 exact YAML/JSON, asset metadata, joint mapping, guard/PD/native limits
3. 1–2 episode의 frozen A/A raw capture CSV 및 schema/readme
4. reference trajectory/phase clock 생성 코드와 Viser preview CSV
5. latest result report와 failure receipt

이 다섯 묶음으로 frozen authority audit은 시작할 수 있다. 그 뒤에만 큰 checkpoint collection이나 raw rollout를 받는다.

