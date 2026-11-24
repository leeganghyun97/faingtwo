# 핵심 증거 상태 표

이 표는 서로 다른 policy, parser, asset, runtime을 하나의 성공률로 합치지 않기 위한 최소 비교 지도다. `physical execution`은 당시 기록된 실행 범위이며, 새 컴퓨터에서의 재현을 뜻하지 않는다.

| 계열 / 시점 | source kind·실행 범위 | 직접적으로 말할 수 있는 것 | 말하면 안 되는 것 | authority / comparability |
|---|---|---|---|---|
| Update1700 SafeDirect development cohort | historical frozen evaluation, 20 env × 500 control (10 s) | 원본 parser로 qualified touchdown 400/400, env별 L/R 10/10, strict alternation 20, hard/fall 0/20이 기록됨 | 현재 CReF/Teacher/V18 또는 새 asset에서의 일반화·strict resume 성공 | historical report; parser/asset/runtime and independent seed parity are not established |
| U730 course analysis | historical frozen 100 env × 500 controls | cyclic survival/gait pass 100/100, hard/fall 0, forward displacement 기록 | heading/course control 성공; straightness pass는 0/100 | same report scope only; U1700 parser와 같은 gate 아님 |
| CReF 2k static closure | native run, 15 env × 20 controls × 7 updates | finite training, S0 survival improvement, checkpoint emission | stable forward gait; completed episode mean displacement는 negative | direct native data; short horizon and static-closure task |
| CReF frozen height audit | frozen deterministic/mean-action replay | posture decay → height decay → physical fall → height termination 순서 | original stochastic episode와 bit-identical long-run replication | direct replay for this checkpoint, not cross-lineage proof |
| CReF matched COM diagnostic | matched-state frozen intervention analysis | natural state COM outside individual sole; bounded COM assist → partial unload but no contact loss/swing/TD | all X2 policies의 단일 universal root cause | strongest CReF mechanism, but actuator effort/CoP still limited |
| U17 APA/transfer funnel | later Teacher/support logs | support ready 55,076/61,882; swing unloaded 346; both 5; physical ≥1 mm lift 0; confirmed TD 0 | CReF checkpoint direct performance | same platform/15D lesson, different run/measurement lineage |
| V18 F0/F1/F2 R33/R34 | bounded native capture, 64 env × 100 controls, optimizer 0 | hard events 0 in short exposure; knee-applied ≥1 mm clearance rows 0 | normal stepping, native motor torque/impulse proof, long-run safety | generic side fields mismatch → several deltas `NOT_COMPARABLE`; effort/impulse unknown/not collected |

## Qualification labels

| Label | Definition |
|---|---|
| historical gait evidence | A report of a prior physical/frozen evaluation under its own source, parser, asset and runtime contract. It remains useful evidence but does not certify a new host. |
| static pass | Import/config/tensor/one-step result. No gait claim. |
| safety pass | No hard event in the declared finite budget. No gait or long-horizon claim. |
| physical lift | Must include a declared terrain-relative clearance and support/contact context. Gate is run-specific. |
| qualified/confirmed touchdown | State-machine event defined by a particular run. Cannot be compared by count unless the parser/gates/timebase match. |
| `NOT_COMPARABLE` | Same metric name but different side, prestate, gate, phase, population, or frame. Do not rank candidates using it. |

## Proposed next-experiment gates, not universal constants

`COM margin > +10 mm`, three-control persistence, 1 mm and 5 mm clearance are **proposed next-experiment gates** from this research program. They are not Agibot specifications, universal biomechanics thresholds, or guarantees of safe gait. Every new run must record its own foot geometry, terrain reference, sensor pair, timebase and gate code/version.

