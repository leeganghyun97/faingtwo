# 현재 상태, 확정된 교훈, 다음 의사결정

## 현재 판정

**연속 보행은 아직 검증되지 않았다.** 여러 PPO, reward, PD, reference, privileged-Teacher, CReF/recurrent, SafeDirect 및 Viser 계열을 시도했지만, 최신 검토에서 물리적으로 유효한 `unload → clearance ≥1 mm → touchdown → 새 지지 발 하중 인수 → root-height 회복` 연쇄를 같은 episode/timebase에서 입증하지 못했다.

이 판단은 "모든 실험이 실패했다"는 뜻은 아니다. 다음 기반은 보존할 가치가 있다.

- 15D locomotion action에서 31D target으로 가는 SafeDirect/guard 경로의 분리 관점
- pre/post-physics, reset generation, landing ID, target/q/qd를 분리해 기록해야 한다는 계약
- reference 보상과 실제 contact 전환을 별도 metric으로 판단하는 원칙
- Teacher의 privileged signal과 Student deployable observation을 분리하는 구조
- 체중 이전(stance share, swing Fz, COM/support relation)을 lift보다 앞선 gate로 다루는 방향

## 반복적으로 관측된 병목

| 병목 | 근거 수준 | 의미 | 다음에 먼저 확인할 것 |
|---|---|---|---|
| 스윙 발 언로딩 부족 | 반복 관측 | `support_ready`가 많아도 `swing_unloaded`와 `transfer_ready`가 거의 성립하지 않았다 | foot body/pair/Fz 단위·freshness, stance share와 swing Fz의 같은 substep 정렬 |
| lift metric과 실제 지지 전환의 괴리 | 반복 관측 | μm~mm rise 또는 reference 추종이 true swing을 보장하지 않았다 | terrain-relative sole clearance, swing contact release, 반대발 지지와 함께 기록 |
| reward hacking | 반복 관측/설계 감사 | 양발 지지·진동·미세 자세 변화·낙상 직전 상태가 lift/forward/touchdown 관련 return을 얻는 경로가 있었다 | raw/weighted/dt contribution 및 event identity를 action/physics와 같이 로그 |
| reference-target-q gap | 관측됨, 원인은 미확정 | reference가 있어도 SafeDirect target 부족·projection·tracking·contact 제약 중 어디서 사라지는지 별도 분해가 필요하다 | reference→raw action→effective action→approved target→q/qd→foot FK trace |
| sensor/force/CoP authority 부족 | 반복 미측정 | native contact impulse, friction impulse, contact point/CoP, motor-only torque가 일부 runtime에서 없거나 의미가 불확정이었다 | API/buffer/pair identity/timebase를 physical conclusion 전에 검증 |
| 비교 설계 오류 | 관측됨 | 좌우 스윙 side가 다른 trace를 같은 발 geometry delta로 비교한 사례가 있어 `NOT_COMPARABLE`로 처리됐다 | A/A repeat, common prestate, 같은 anatomical foot, seed만이 아닌 state 확인 |

## 다음 실험의 순서

다음 순서를 건너뛰고 PPO reward/PD gain/action scale만 키우지 않는다.

1. **F0/F1 frozen authority capture**: 기존 baseline을 action 변경 없이 A/A 두 번 평가한다. foot body, contact pair, Fz, q/qd/target, terrain-relative sole clearance, whole-body COM, root/torso, reset generation, pre/post timestamp를 확인한다.
2. **작은 단일 A/A/B 물리 개입**: lateral weight-transfer *또는* knee target authority처럼 하나만 적용한다. 항상 guard와 native limits를 유지한다.
3. **성공 metric 판정**: `stance_load accepted`, `swing force decreases/release`, `clearance ≥1 mm`, `touchdown`, `post-touchdown height recovery`를 같은 landing ID로 연결한다.
4. **5-update Teacher canary**: 1–3에서 실제 반응을 본 후에만 시작한다. PPO reward 평균이나 visual plausibility만으로는 통과가 아니다.
5. **Teacher qualification 후 Student**: Teacher가 independent frozen evaluation에서 반복 성공하기 전 BC/DAgger/Student PPO를 시작하지 않는다.

## 최신 V18 bounded exposure의 한계

V18 F2 R33/R34는 bounded no-step hard-safety exposure이며 optimizer step은 0이었다. 64 env × 100 controls의 capture에서 knee-applied physical clearance ≥1 mm row는 0이고, 일부 lateral-only reference와 knee candidate는 계획된 swing side가 달라 generic geometry delta가 `NOT_COMPARABLE`이다. native effort 및 contact impulse도 `UNKNOWN/NOT_COLLECTED`였다. 따라서 이 결과를 보행·native torque·contact impulse의 증명으로 사용하면 안 된다.

## 새 컴퓨터에서 Codex가 지켜야 할 불변 규칙

- raw Gaussian action, feasible/applied action, native target readback, q/qd를 하나의 값처럼 취급하지 않는다.
- software PD proxy는 native motor torque가 아니다.
- force × dt를 measured impulse라고 부르지 않는다.
- post-step target readback은 pre-step solver consumption의 증거가 아니다.
- foot-center Fz weighted average는 contact point/wrench authority 없이는 ground-truth CoP가 아니다.
- Student actor 입력에 training-only privileged physics/contact/load/COM이 들어가면 안 된다.
- `PASS`가 import/smoke/safety/finite tensor인지 gait qualification인지 항상 범위를 명시한다.
- checkpoint observation 차원이 다른 계열을 `strict=False`로 억지 load하지 않는다. 특히 V18 P0 U5는 source/contract의 outer 948D·critic 811D와 checkpoint tensor의 87D core·804D critic이 아직 하나의 strict-load identity로 닫히지 않았다. native physics 없이 original import graph에서 strict-load receipt를 먼저 만든다.
