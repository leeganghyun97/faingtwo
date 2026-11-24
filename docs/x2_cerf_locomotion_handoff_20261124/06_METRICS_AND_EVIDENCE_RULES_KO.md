# 측정값, 용어, 증거 해석 규칙

## 보행 성공을 판정하는 최소 연쇄

한 control에서 보이는 발 높이 또는 reward만으로 보행을 선언하지 않는다. 같은 `episode generation`, `landing_id`, `foot side`, `control/substep timebase` 안에서 아래 연쇄를 추적한다.

\[
\text{stance contact/load} \rightarrow
\text{COM/support preparation} \rightarrow
\text{swing Fz decrease or release} \rightarrow
\text{terrain-relative clearance} \rightarrow
\text{touchdown} \rightarrow
\text{new stance load acceptance} \rightarrow
\text{root-height recovery}.
\]

권장 초기 physical gates:

| Gate | 최소 증거 |
|---|---|
| G0 | joint/body/contact mapping, timestamp, reset generation authority |
| G1 | 원래 지지 발이 유효하고 stance share 또는 support margin이 개선됨 |
| G2 | swing foot의 Fz 감소/release와 terrain-relative sole clearance ≥1 mm가 같은 이벤트에서 관측됨 |
| G3 | clearance ≥5 mm, root/torso stability 및 guard safety 유지 |
| G4 | 새 발 접촉, 지속적인 load acceptance, 중복 보너스 없는 touchdown |
| G5 | 좌우 교대가 반복되고 전진 거리·heading·height recovery가 유지됨 |

## metric의 정확한 의미

| 이름 | 의미 | 흔한 오해 |
|---|---|---|
| `sole_rise_mean_mm` | 기록 정의에 따른 기준 높이 변화의 평균 | terrain-relative clearance 또는 contact release를 보장하지 않음 |
| `active_swing_sole_rise_mean_mm_per_env` | active swing으로 판정된 env/control의 평균 rise | 비활성/종료 env를 제외하므로 전체 평균보다 클 수 있음 |
| `physical_lift` | run-specific physical gate를 통과한 lift | gate 정의가 바뀌면 run간 직접 비교 불가 |
| `qualified/confirmed touchdown` | landing state machine의 진단 event | 별도 bonus 지급 판정기와 동일 event count로 합치면 안 됨 |
| `stance_share` | \(F_{z,stance}/(F_{z,L}+F_{z,R})\) | Fz body/pair/freshness가 검증되지 않으면 physical truth로 결론 내릴 수 없음 |
| COM | whole-body mass-weighted center of mass | root position 또는 pelvis position과 동일하지 않음 |
| CoP | contact wrench/contact point에서 산출한 pressure center | 양발 중심의 Fz 가중 평균은 ground-truth CoP가 아님 |
| PD proxy | \(K_p(q_{target}-q)-K_d\dot q\) 형태의 software 진단값 | native actuator motor torque가 아님 |
| target–q gap | approved/native target과 measured q의 차이 | target 생성 부족, rate projection, tracking, contact constraint를 구분하지 못함 |

## 보상 해킹을 차단하는 검사

다음 패턴은 reward 증가가 보행 개선을 뜻하지 않을 수 있다.

- 양발 접촉·static pose를 유지하며 bilateral/dwell/phase return만 반복 획득
- micro-vibration으로 clearance/velocity delta/threshold crossing을 반복 획득
- root 자체가 내려가거나 몸을 던져 foot height/progress를 만드는 방식
- touchdown 직후 recovery gate가 꺼져 support takeover를 보상하지 못하는 방식
- reference measured-q reward를 standing pose에서도 크게 얻는 방식
- total Fz 감소를 transfer/unload 성공으로 오인하는 방식

각 보상은 raw quantity, gate, weight, dt handling, clipping, per-control contribution, landing/episode identity와 함께 기록해야 한다. dense term에 dt를 두 번 곱하거나 one-shot event를 rate화하지 않고 매 control 지급하는 일도 피한다.

## `NOT_*` 표기

| 표기 | 의미 |
|---|---|
| `NOT_RUN` | 실행하지 않았다. 0 또는 실패와 다르다. |
| `NOT_MEASURED` | 물리 API/log가 없거나 authority를 검증하지 못했다. 추정값으로 대체하지 않는다. |
| `NOT_COMPARABLE` | side, prestate, timebase, gate 또는 population이 달라 정량 비교가 무효다. |
| `UNKNOWN` | 값은 보이지만 source/units/meaning을 검증하지 못했다. |

## CReF/Teacher/Student 경계

HDMI MLP PPO, HDMI CReF teacher recovery, RoboJuDo-Plus recurrent CReF, 역사적 walk successor는 서로 다른 소스와 관측·runtime 계약을 가진다. CReF의 recurrent actor 혹은 privileged critic을 가지고 있다고 해서 X2 physical gait가 성립한 것은 아니다. Student의 actor에는 실제 배포 가능한 proprioception, IMU, depth, command, history만 넣는다. ground-truth velocity, contact/load, COM, terrain truth는 training-only privileged channel 또는 auxiliary target로만 쓴다.

