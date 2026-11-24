# 원본 자료를 추가로 받을 때 보내는 요청 양식

아래 템플릿을 원본 작업공간 보유자에게 보낸다. 파일 하나만 받아도 provenance가 끊기지 않도록 작성했다.

```text
X2 CReF locomotion 자료 요청

목적: [F0/F1 frozen authority audit | single A/A/B intervention | strict resume | reference inspection]
run/branch ID:
checkpoint path and SHA-256:
source commit and dirty-worktree status:
Python executable and Isaac/Lab version:
GPU driver/CUDA/PhysX version:
asset URI/path + URDF/USD/mesh/collision variant hash:
control dt / physics dt / decimation:
num_envs / rollout controls / seed / terrain / command:
15D policy joint names in actor order:
31D target joint names in articulation order:
SafeDirect guard / target rate / position / velocity / effort limit source:
PD profile source and native readback availability:
observation layout and normalization state:
reward/phase/curriculum config and progress state:
recurrent state contract (if any):
exact evaluation command:

For each requested log, include:
- episode generation, env ID, control ID, physics substep and pre/post timestamp
- raw/effective action, approved target, q/qd
- foot body/contact pair/Fz and unit/source authority
- terrain-relative sole clearance and root/torso/COM source
- reset/terminal/landing ID and reward component raw/weighted/dt contribution

Do not replace unavailable native effort, contact impulse, CoP or contact point with a proxy. Mark it NOT_MEASURED.
```

## 파일 크기를 줄이는 방법

- 모든 env의 full trace보다, frozen A/A baseline 두 번의 event window와 metadata를 우선한다.
- 값이 일정한 static config는 한번만 보낸다.
- float32 CSV보다 parquet 또는 gzip CSV를 쓰되 schema/units을 함께 보낸다.
- checkpoint는 selected baseline 한 개만 보내고 SHA-256 및 normalization/progress sidecar를 같이 보낸다.
- media/video는 diagnosis가 끝난 뒤 필요하면 추가한다. Viser screenshot은 physical evidence를 대체하지 않는다.

