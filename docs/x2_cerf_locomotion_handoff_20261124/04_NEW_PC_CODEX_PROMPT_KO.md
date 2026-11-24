# 새 컴퓨터의 Codex 시작 프롬프트

아래 블록을 새 컴퓨터에서 이 폴더를 연 뒤 Codex에 그대로 붙여 넣는다. 원본 대형 자료가 없는 상태에서도 과장 없이 다음 진단을 시작하도록 만든 프롬프트다.

```text
이 디렉터리는 X2 CReF locomotion의 문서 기반 인계 패키지다. 먼저 아래 파일을 모두 읽고, 서로 다른 소스 체계와 증거 범위를 혼동하지 마라.

1. README_KO.md
2. 00_CURRENT_STATE_AND_DECISION.md
3. agent_drafts/01_METHODOLOGY_AND_EXPERIMENT_HISTORY.md
4. agent_drafts/02_RUNTIME_REPRODUCTION_AND_COMMANDS.md
5. agent_drafts/03_DIAGNOSTICS_FAILURES_AND_ARTIFACT_MAP.md
6. 06_METRICS_AND_EVIDENCE_RULES_KO.md
7. source_maps/ORIGINAL_EVIDENCE_MAP.md
8. agent_drafts/06_LINEAGE_DIMENSION_VERIFICATION.md

목표는 reward scale을 임의로 올리는 것이 아니라 X2가 실제로 체중을 지지 발로 옮기고 반대발을 언로딩해 1 mm 이상 terrain-relative clearance를 만든 뒤 touchdown과 새 지지 인수를 수행할 수 있는지를 검증하는 것이다.

**현재 폴더가 문서-only handoff라면, simulator/PPO/PD/guard/controller 실행 명령을 제안하거나 실행하지 마라. 먼저 inventory, missing-input request, capture contract, A/A/B plan만 작성한다.** 실행은 original source·asset·checkpoint·runtime·native authorization이 별도로 확인된 뒤에만 다룬다.

먼저 다음을 수행해라.

A. 현재 새 컴퓨터에서 사용 가능한 소스, checkpoint, URDF/USD/mesh, Isaac Sim 버전, Python/Conda env, GPU/driver, native contact API를 inventory로 작성한다.
B. 원본 실행물이 없는 항목은 추측으로 재현했다고 하지 말고 MISSING으로 적는다.
C. `F0/F1 frozen A/A` 평가 계약을 작성한다. action 변경 없이 다음 값을 같은 control/substep 기준으로 확보해야 한다: actor raw/effective action, approved 31D target, q/qd, foot pose/linear/angular velocity, terrain-relative sole clearance, foot/body contact pair, Fz, whole-body COM/velocity, root/torso pose, episode generation/reset, pre/post physics timestamp.
D. Fz, pair identity, contact point, impulse, native motor effort의 API 의미와 단위·timebase를 검증한다. 없는 값은 NOT_MEASURED로 둔다. foot-center weighted Fz를 CoP ground truth로 사용하지 않는다.
E. F0/F1이 반복 가능하면 only one intervention A/A/B를 준비한다. 후보는 (i) 제한된 lateral weight transfer 또는 (ii) knee target authority 중 하나만 선택한다. PD gain, guard, native limits, asset, reward를 동시에 바꾸지 않는다.
F. 성공은 다음의 동일 landing ID/episode에서만 판단한다: stance load acceptance → swing Fz 감소 또는 release → terrain-relative clearance ≥1mm → touchdown → new stance load acceptance → root height recovery. reference tracking, reward 상승, Viser 자세, micro-lift만으로 성공 선언하지 않는다.

엄격한 규칙:
- SafeDirect guard/native position/velocity/effort limit을 유지한다.
- software PD proxy와 native motor torque를 구분한다.
- `PASS`의 종류(import, static, smoke, safety, physical gait)를 보고서에 분리한다.
- student actor의 observation에는 privileged contact/load/COM/ground-truth velocity를 직접 넣지 않는다.
- 기존 CReF recurrent source와 HDMI MLP/teacher source를 같은 runtime으로 가정하지 않는다.
- 803/809/1612, 85D+depth/802, 933D, 948D, P0 U5의 87D core/804D는 서로 다른 또는 미해결 observation contract다. strict-load receipt 없이 cross-resume하지 않는다.

답변 전에 `NEW_HOST_INVENTORY.md`, `F0_F1_CAPTURE_CONTRACT.md`, `MISSING_INPUTS.md`, `FIRST_AAB_PLAN.md` 초안을 작성하고, 실제 Isaac 또는 PPO 실행은 필요한 source/asset/checkpoint의 존재와 one-shot safety preflight가 확인된 경우에만 제안해라.
```

## 원본 실행물을 나중에 받았을 때 사용할 추가 프롬프트

```text
원본 X2 runtime, asset, checkpoint 또는 raw log를 추가로 받았다. 먼저 SHA-256, source commit, Python executable, Isaac version, asset metadata와 joint-name mapping을 기록하라. 이후 immutable F0/F1 frozen A/A capture를 실행하고, existing reward/PD/guard/limit/controller를 변경하지 말라. A/A repeatability와 contact measurement authority가 통과한 뒤에만 단일 B intervention을 설계하라.
```
