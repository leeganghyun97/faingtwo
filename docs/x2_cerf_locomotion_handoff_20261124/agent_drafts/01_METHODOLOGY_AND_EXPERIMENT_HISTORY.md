# X2 CReF Locomotion — 방법론·실험 이력 인수인계

작성일: 2026-11-24 (정적 원본 검토만 수행)  
범위: `/home/fain/Desktop/HDMI` 및 이관 archive에 남은 X2/CReF 보행 실험 기록.  
비범위: 이 문서는 Isaac 물리, PPO, checkpoint 역직렬화, W&B 재개를 실행하지 않았다.

## 0. 이 문서를 읽는 방법

이 프로젝트에는 서로 다른 시기·소스·관측 차원을 가진 학습기가 공존한다. 폴더 이름의 `U100`, `U730`, `V18`, `PASS`, `COMPLETE`만으로 같은 모델 계보나 보행 성공을 뜻하지 않는다. 새 컴퓨터에서는 다음을 항상 함께 고정한다.

1. 실행 source/commit과 Python·Isaac 버전
2. 로봇 asset, 31-D articulation joint order, 15-D locomotion action order
3. observation layout, action scale, SafeDirect/guard/PD profile
4. phase clock, reward dt 계약, reset/termination 정의
5. parent checkpoint SHA, strict resume인지 warm start인지
6. 실행 manifest, trace, frozen evaluation receipt

이 문서의 숫자는 각 원본 보고서에 남은 값이다. 이관 검토가 과거 결과를 새로 재현한 값이 아니다. `NOT_RUN`, `PENDING`, `UNKNOWN`, `NOT_MEASURED`, `NOT_COMPARABLE`은 0 또는 성공으로 바꾸지 않는다.

## 1. 코드·학습기 계보

| 계열 | 보존된 주 소스 | actor/critic 및 관측 | 인수 시 핵심 주의점 |
|---|---|---|---|
| HDMI tracking/LMO | `active_adaptation/`, `cfg/task/X2/`, `scripts/` | 기본 actor는 MLP Gaussian, critic은 policy+privileged 입력 | 이전 action을 observation에 추가한 것만으로 GRU가 되지 않는다. |
| HDMI CReF 2603 teacher 복구 | `cfg/task/X2/cref2603/teacher.yaml`, `cfg/algo/cref2603_ppo.yaml` | progress goal, height scanner, signed safety reward | 삭제된 `unitree_rl_lab`의 일부를 복구한 계열이다. 원본 전체가 아닐 수 있다. |
| RoboJuDo CReF | 별도 sibling `../RoboJuDo-Plus/robojudo/cref/` | metric depth CNN → velocity LSTM → attention → temporal GRU → 15-D action, asymmetric critic | HDMI 안의 동명 vendor 폴더와 구분한다. source import 순서가 실행 identity를 바꾼다. |
| 역사적 walking successor | 각 `walk_*`, `x2_*` 아래의 독립 `source/`, launcher, runtime | CReF recurrent actor/adapter/SafeDirect와 reward·reference 변형 | 동명 모듈이 많으므로 각 run의 source SHA와 `RESOLVED.json`/manifest가 권위다. |

### X2 물리·제어 공통 계약

- asset 등록은 `active_adaptation/assets/x2.py`가 중심이다. 전체 URDF/USD, simple-collision 변형, HDMI URDF/MJCF는 외부 `Agibot-x2_stand_train` 의존성을 가진다.
- URDF는 actuated joint 31개, locomotion은 legs+waist 15-D, tracking은 legs+waist+shoulders+elbows 23-D를 사용한 이력이 있다. 이 집합을 섞으면 checkpoint를 의미 있게 재개할 수 없다.
- 후속 native 계열은 control 20 ms, physics 5 ms, decimation 4를 사용했다. reward에 dt가 이미 적분됐는지 run별 reward 코드에서 확인해야 한다. one-shot reward는 `1/dt`로 rate화한 뒤 한 번만 지급하는 계약이 있다.
- `Kp(q_target-q)-Kd qd`는 software PD proxy다. native motor-only torque가 없으면 이것으로 torque saturation을 판정하지 않는다.
- SafeDirect 계열은 raw Gaussian action, guard 통과 effective/applied action, native target readback, measured q/qd를 분리해 기록한다. PPO ratio는 clipped target이 아니라 저장한 latent Gaussian action/log-prob를 사용해야 한다.

## 2. CReF/PPO 구조와 재개 시점의 함정

### HDMI 기본 PPO/CReF

기본 `PPOPolicy`는 actor `[512, 256, 256]` MLP Gaussian, critic `[256, 256, 128]` MLP다. 기본값은 `train_every=32`, epoch 3, minibatch 8, Adam lr `1e-4`, ratio clip `0.2`, entropy `0.001`, GAE `gamma=.99`, `lambda=.95`다. CReF 2603 config는 negative reward clamp를 `null`, initial noise `.2`, compile false로 둔 이력이 있다. 이는 signed safety penalty를 GAE에 남기는 목적이며 보행 수렴 증거가 아니다.

복구 teacher의 policy observation은 803-D (`command4 + progress4 + angular velocity3 + gravity3 + q15 + qd15 + previous action45 + height scan714`), critic용 priv는 809-D다. 기존 critic concat 입력은 1,612-D다. native action scale은 waist `.12`, hip yaw/roll `.20`, hip pitch `.22`, knee `.28`, ankle `.18` rad/output-unit으로 기록되어 있다. 이 unbounded Gaussian joint-position 경로와 SafeDirect 제한 경로를 동일한 제어기로 표현하면 안 된다.

### RoboJuDo recurrent CReF

외부 recurrent actor는 metric optical depth `[N,1,48,64]`를 4 m 기준으로 normalize하고 CNN으로 depth token을 만든다. proprio와 mean depth feature를 velocity LSTM(64)에 넣어 3-D body velocity를 추정하고, depth tokens에 attention을 건 뒤 temporal GRU(128) 및 highway gate를 거쳐 15-D Gaussian mean을 낸다. critic에는 proprio+GT body velocity 3-D+robot-centric terrain이 들어간다. GT velocity는 auxiliary MSE target이지 actor input leakage가 아니어야 한다.

`isaac_ppo.py` 계열은 `[T,N,…]` rollout과 contiguous TBPTT chunk를 사용한다. hidden state는 done env만 초기화하고 chunk 시작 memory는 detach한다. timeout은 reset 전 `terminal_observation`으로 bootstrap하고 true termination은 0 bootstrap이며, 둘 다 episode 간 GAE recursion을 끊는다. checkpoint는 network/Adam/config/counters를 저장하지만 simulator state·RNG의 완전한 물리 continuation은 보장하지 않는다. CPU MuJoCo runner에는 collection 시작 시 episode를 재개하는 별도 계약이 있으므로 모든 CReF runner의 시간 특성을 하나로 일반화하면 안 된다.

## 3. 시간순 연구 이력과 확인된 결론

| 시기/계열 | 시도한 핵심 방법 | 원본에서 확인된 결과 | 다음에 남긴 교훈 |
|---|---|---|---|
| 2026-10 초반 CReF 복구·정적/terrain smoke | CReF config/runner 복구, 8/128 env smoke, terrain tile | finite tensor, reset, checkpoint scheduling의 일부가 확인됐으나 보행 수렴은 증명되지 않았다. 15 env×T20×50 update의 15,000 transition도 finite-training 시험이다. | runtime PASS와 terrain traversal/보행 성공을 분리한다. |
| 10/6–17 외부 WALK1A/CReF | directional/dual expert/phase/actuator 후보 | progression U20은 남았지만 Level2=0, BC/PPO 새 step=0인 보고가 있다. vendor 준비 상태는 외부 제출 증거가 아니다. | motor-only effort/headroom 미관측을 actuator 불가능성으로 확대하지 않는다. |
| 10/17–18 direct locomotion/physical credit | resumed locomotion, direct PPO, contact-loss credit 수정 | resume은 생존 20/20이지만 forward walk 0/20, alternating TD 0. direct PPO는 logical update 436, optimizer 6,976, transition 3,571,712인데 qualified TD 0, hard safety 1. Stage0 matched A/B deterministic sole rise 0. | 대량 transition·생존·return은 swing/landing 성공의 대체 지표가 아니다. |
| 10/18–28 guard/PD/native lifecycle | PD demand guard, q/qd/target, action/rate/position intersection, reset/termination ledger | U99/U102에서 native writer/reader, pre/post physics, selected reset, recorder와 PPO lifecycle를 분리했다. update 100–102는 training transition 24,576/optimizer 48과 discarded partial 2,688을 구분했다. strict swing 13·qualified TD 12가 있어도 alternation 0이었다. | raw action→target→q/qd→foot/contact의 각 경계를 계측한다. |
| 10/29–30 SafeDirect/직진·자연스러움 | env-local termination, SafeDirect, stride/knee/straightness | 역사적 Update1700 20-env×10 s 개발 평가에는 qualified TD 400, 환경별 L/R 각 10, strict alternation 20, fall/hard 0, mean vx 0.1671 m/s가 기록됐다. | 가장 강한 historical gait evidence지만 development cohort이며 독립 seed 일반화는 별도다. |
| 10/31–11/5 perception/terrain/MoE/uphill | native RGB-D+LiDAR, stair curriculum, terrain expert/router, ramp posture | MoE는 flat parity는 좋지만 0.75° sensor-selected MoE parity가 충족되지 않았다. R53 U25→U30도 5 update 후 gait 3/8→2/8, forward .9219→.8827 m로 악화했다. | flat 성능을 slope/generalization으로 부르지 않는다. low-slope viewer grid도 simulator terrain mesh의 재현이 아닐 수 있다. |
| 11/6–11 landing/U100–U730 | lateral adapter, impact ledger, asymmetric ankle-roll, phase/pose reference, U230→U730 | U730은 500 PPO update/3.2M transition/optimizer 7,000을 완료했고 frozen 100 env×10 s에 gait 100/100, hard/fall 0이지만 straightness 0/100, heading RMS 13.709°였다. | gait survival, forward travel, heading/cross-track, confirmed landing을 따로 평가한다. reward mean/straightness activation 상승만으로 course control 성공으로 승격하지 않는다. |
| 11/12–19 lift·PD·task-space | lift multiplier/ladder, bilateral dwell, one-leg support, stiffness/PD candidate, task-space touchdown anti-hack | 여러 관절·reward 후보에서 low rise 또는 reward hacking이 반복되었다. 폴더의 `10k`, `1500` 같은 이름은 완료 update 수가 아니다. | threshold/scale을 계속 올리는 것보다 지원 하중·COM·action authority를 분해하는 것이 우선이다. |
| 11/17 Teacher/Student | privileged physics Teacher, deployable depth/proprio Student, BC/DAgger gate | Teacher V8은 5 PPO update/10 optimizer/5,120 transition, local termination 22, hard safety 0, U5 checkpoint만 기록됐다. 필수 20-env frozen eval은 `NOT_RUN`; BC/DAgger/Student PPO는 `NOT_RUN`이다. | Teacher training 완료와 Student deployability는 별도 qualification이다. |
| 11/20 U17 APA/weight transfer | Fz, COM, CoP authority audit; planned lateral shift vs knee target authority | active swing 61,882 중 support ready 55,076 (89%), swing Fz ≤10 N 346 (0.559%), 동시 5 (0.00808%), physical lift ≥1 mm 0, confirmed TD 0. | 실제 병목은 support readiness보다 unloading/weight transfer 결합이다. true CoP/motor torque는 `NOT_MEASURED`. |
| 11/22–23 safe landing/policy prior/V18 support-first | imitation authority, contact takeover, support-first FSM, frozen A/A/B | V17은 lift가 있어도 negative root-forward/height loss가 많았다. V18 frozen A/A는 repeatable하지만 all 32 `lmo_base_height_below`, TD 0. 29→33 control phase-window shadow가 timing mismatch를 보였다. | reward window·support transfer·posture recovery를 분리한다. offline rescoring은 새 물리 행동이 아니다. |
| 11/24 R33/R34 audit | lateral/ankle/knee small intervention A/A/B | generic swing/stance field가 anatomical side mismatch라 이전 headline 하중 변화가 과장됐음을 발견했다. knee-applied row의 ≥1mm clearance는 0이다. | side-specific left/right raw fields로 비교한다. `PASS`는 2초 bounded hard-safety exposure일 뿐 stepping 성공이 아니다. |

## 4. 중요했던 성공과 성공으로 부르면 안 되는 것

### 역사적으로 가장 강한 gait evidence: Update1700

원본 `walk_update1000_safe_direct_envlocal_100env_to3000_20261029/SWING_TOUCHDOWN_VALIDATION_REPORT.md`는 Update1700에 대해 다음을 보고했다.

- 20 평가 환경 모두 500 controls=10 s에서 qualified touchdown 20회
- 환경별 left/right 10/10, strict alternation 20, 전체 400/400 qualified, unqualified 0
- fall/hard failure 0/20
- swing air duration median 22 controls=0.44 s
- sole clearance 상승 최소 12.30 mm, 평균 28.21 mm
- 반대발 support fraction 1.0, recontact Fz 최소 194.62 N
- 평균 vx 0.1671 m/s, vx RMSE 0.02156 m/s
- checkpoint SHA-256 `ab1bd0f0233b48b537a868b66973d62acf46887c4d500a78fc317b088b70abd7`

여기서 **qualified touchdown**은 이 원본 독립 parser의 엄격한 정의를 뜻한다. Air 상태, liftoff 이후 5 mm 이상 상승, 반대발 지지 80% 이상, Fz 10 N 이상 2 controls 재접촉, 동일 control 양발 landing 제외, hard-safety 이벤트 제외가 포함된다. 이후 실험의 `qualified`, `confirmed`, `physical lift`는 같은 판정기라는 보장이 없다.

이 결과는 강한 development-cohort evidence지만, 새 컴퓨터의 새로운 seed·asset SHA·runtime과 동등함을 증명하지 않는다. 그대로 "현재 가장 좋은 일반화 보행 정책"이라고 부르지 않는다.

### U730: 보행 유지와 직진성의 분리

`x2_u230_course_scope_repair_u730_analysis_20261109/U730_FINAL_RESULT.md`의 frozen 100 env×500 controls 결과는 gait pass 100/100, hard/fall 0/100, survival 10.0 s, forward displacement 1.5366 m, efficiency .8650, alternating TD 최대 19를 기록했다. 하지만 straightness pass 0/100, heading RMS 13.709°, mean absolute final heading 21.176°, cross-track RMS .1074 m였다. 이 문서는 **walking-like cyclic survival**과 **course/heading control**이 다른 목표임을 보여준다.

### lift가 관측됐지만 stepping은 아니었던 후보

`x2_safe_landing_teacher_il_u20_v8_reference_hold_measurement_recovery_20261122/FINAL_REPORT.md`의 U5 imitation authority canary는 full feasible imitation c0에서 ≥1 mm lift 6.67%, ≥5 mm lift 4.89%, max rise 14.08 mm, qualified touchdown 10, confirmed touchdown 0을 기록했다. strong feasible imitation c1은 ≥1 mm .44%, ≥5 mm .11%, TD 0/0이다. c0도 negative root-forward와 height loss가 있어 stable stepping이 아니다.

V17 C0 U33/C1 U27은 조건부 physical-lift 약 3.4%, conditional rise 약 28–29 mm였지만 전체 control 평균 rise는 약 1.6 mm, root forward는 약 −.0045 m/control, height/fall termination이 많았다. 조건부 lift 평균을 지속 보행 clearance로 바꾸어 해석하면 안 된다.

## 5. 보상·reference·teacher의 발전과 반복된 병목

### 5.1 Reference gait와 task-space prior

`x2_reference_gait_formula_preview_20261116`는 minimum-jerk curve

\[
M(s)=10s^3-15s^4+6s^5
\]

로 연결한 C2 reference를 썼다. 좌우는 반주기 차이이며 knee peak phase .42가 hip peak phase .72보다 앞선다. 이 preview/IK는 시각화 reference이지, native target write나 stable gait의 증거가 아니다. 절대 q와 standing-relative offset, hip/knee sign, phase ordering을 run별로 재검증해야 한다.

후속 user pose/U230/U730 계열은 hip/knee/ankle/waist/arm prior, knee timing, arm swing, stiffness, action scale, rollout phase와 touchdown logic을 조합했다. 유용한 기여는 target–q gap과 motion-order 관찰을 남긴 점이지만, reference를 잘 따르는 reward가 실제 unilateral unloading을 보장하지는 않았다.

### 5.2 Reward hacking에서 배운 규칙

과거 실험에서 다음 보상 경로가 실제 보행과 분리될 수 있음이 반복됐다.

- phase prior나 teacher angle similarity만으로 reward가 발생하면, foot clearance가 없는 정지 자세·미세 진동도 유리해질 수 있다.
- lift threshold를 매우 낮추거나 1 mm에서 포화시키면 μm 수준 흔들림을 보상하거나 1 mm 이상 추가 행동을 유도하지 못한다.
- forward/lift reward를 직접 크게 주면 한발로 몸을 던지거나 root height를 잃으면서 비정상 touchdown/forward signal을 만들 수 있다.
- touchdown bonus가 접촉 직후 사라지거나 여러 gate의 곱으로 거의 0이 되면 post-touchdown load acceptance/height recovery를 배우지 못한다.
- bilateral dwell penalty가 APA/weight-transfer 준비 구간에도 걸리면 안전한 transfer 전에 episode/credit을 잃는다.
- generic side field로 left/right 비교하면 baseline와 candidate가 다른 anatomical foot을 비교하는 오류가 생긴다.

따라서 현재 권장되는 reward 순서는 `support transfer → unload → clearance → forward swing → touchdown → new-foot load acceptance → root-height recovery`다. later-stage reward가 earlier physical gate 없이 먼저 열리면 reward hacking 가능성이 높다.

### 5.3 V18 support-first potential

V18 설계는 support sign `s`와 half-step 시작 `t0`에서

\[
\Delta y_s=s(y_{COM}(t)-y_{COM}(t_0)),\qquad
\Delta v_s=s(\dot y_{COM}(t)-\dot y_{COM}(t_0))
\]

를 기록한다. stance share, swing Fz, COM displacement/velocity, height/posture를 결합한 support quality `Q_S`를 만들고, lift와 forward는 그 뒤에만 연다.

\[
Q_L=Q_S\sigma(c_{swing}),\qquad
Q_F=Q_L\sigma(\Delta x_{swing}),\qquad
\Phi=.5Q_S+.3Q_L+.2Q_F
\]

\[
r_t=\min(r_{max},[\Phi_t-\max_{\tau<t}\Phi_\tau]_+)
\]

frontier는 measured half-step마다 한 번만 초기화한다. 동일 posture/contact의 반복 진동은 high-water를 넘지 못하므로 보상되지 않는다. 이는 방법 설계/계약이며, 자체로 physical efficacy가 입증된 것은 아니다.

### 5.4 Teacher/Student의 역할 제한

11/17 privileged Teacher는 기존 94-D에 training-only physics 824-D 및 expert reference 15-D를 더한 933-D actor observation을 사용했다. q/qd 31, root state, mass-weighted COM, prior-control foot-ground Fz, terrain, hip/knee/waist expert corridor를 읽고 zero-init privileged correction ±.1을 SafeDirect 앞에 둔다. Teacher reward는 measured-q 기반 `3*r_ref`이며 target 자체를 reward input으로 쓰지 않는 구조다.

Student는 deployable 87-D proprio+metric depth 48×64만 사용하고 simulator course/contact/load 7-D를 제거한다. contact/load/COM auxiliary heads는 action path에서 detach한다. camera mount/latency, IMU frame/bias, odometry는 아직 미검증이다. Teacher가 independent frozen evaluation에서 lift·alternation·height recovery·progress·safety·clean exit를 만족하기 전에는 BC/DAgger/Student PPO를 시작하지 않는 것이 계약이다.

## 6. Contact, COM, CoP, PD에서 확보된 사실과 미확보 사실

### 확립된/활용 가능한 계측

- 일부 run의 Fz는 `cref_{left,right}_foot_ground_contact.data.force_matrix_w[...,2]`에서 온 world-z normal pair force다. vertical load/unloading 지표로 쓸 수 있다.
- whole-body COM은 `body_com_pos_w`와 per-body mass의 mass-weighted average로 계산한 이력이 있다.
- V18 frozen capture에서는 sensor Fz와 observer normal-pair force 오차가 최대 `4.20e-05 N`였고, 720 pre/post physics interval alignment와 A/A byte-identical trace가 통과했다.
- target–q gap, qd, target-rate/position headroom, phase/episode generation은 native target writer/reader 및 run별 ledger로 분리할 수 있다.

### 미확보/절대 대체하지 말아야 하는 계측

- true CoP: contact point와 complete wrench/moment가 없으면 foot center 또는 Fz-weighted foot center를 true CoP라고 부르지 않는다.
- contact/friction impulse: force×dt를 solver-measured impulse로 표기하지 않는다.
- native motor torque: `get_dof_actuation_forces()`가 implicit position actuator에서 0을 반환한 경우에는 torque=0 결론을 내리지 않는다. `UNKNOWN_UNAVAILABLE` 또는 `NOT_MEASURED`다.
- U17별 sensor body/pair identity와 substep freshness는 일부 자료에서 `NOT_MEASURED`다.
- post-step target readback은 pre-step solver consumption의 증거가 아니다.

### U17 병목의 정량 근거

`x2_u17_weight_transfer_closure_20261120/FINAL_REPORT.md`의 3,000-env on-policy trace 분석은 다음 funnel을 남겼다.

| 조건 | active swing 대비 |
|---|---:|
| active swing | 61,882 (100%) |
| stance Fz ≥100 N | 55,076 (89.00%) |
| swing Fz ≤10 N | 346 (0.559%) |
| 두 조건 동시 | 5 (0.00808%) |
| physical sole lift ≥1 mm | 0 |
| confirmed touchdown | 0 |

env 0–15 subset에서 stance Fz 평균은 192.86 N, swing Fz 평균은 209.62 N, median stance share .522였다. intended stance가 평균적으로 더 큰 하중을 받지 않았다. stance-foot-relative COM lateral absolute distance는 평균 140.95 mm이고 phase 뒤로 갈수록 단조 수렴 근거가 없었다. `support_ready`와 `swing_unloaded`의 동시 결핍이 핵심이며, knee tracking만 고쳐도 해결되지 않는다.

동 run에서 reference→native target mean absolute gap은 .1336 rad, native target→q gap은 .0278 rad였다. swing knee peak `+.90 rad`가 target corridor 상한 `+.76 rad`보다 .14 rad 크고 hip reference `−.65 rad`가 corridor lower `−.45 rad`보다 .20 rad 작았다. 즉 **첫 병목은 reference command가 native target까지 도달하지 못하는 authority gap**이고, 그 뒤에 **support transfer deficit**이 별도로 있다. target→q error 50 mrad 이상인 표본도 약 20%이므로 tracking을 무시하지 않되, PD alone을 원인으로 확정하지 않는다.

## 7. 최근 V18 / R33–R34의 가장 중요한 해석 정정

### V18 F0/F1 frozen support capture

checkpoint `c2_moderate_authority_u10` (SHA `65845635fdc8aa7e870fd2d252d889477083ef18f41d203a4f63a46fb6c0578b`)에 대해 동일 seed 32 env×180 control frozen mean-action A/A repeat를 했다. optimizer 0, action/target modification 0이며 trace는 의미 있는 항목이 byte-identical이었다.

- 32/32 `lmo_base_height_below`, median termination control 167
- qualified touchdown 0, confirmed balanced landing 0, hard safety 0
- controls 117–118에서는 stance share .562→.626, swing Fz 176.1→154.9 N
- control 122에서 clearance 4.469 mm와 foot forward 32.418 mm가 관측됐지만 active 29-control window 직후였다.
- control 161–163에서는 clearance 10.91→27.52 mm가 나타나도 root height .556→.529 m, support-foot margin negative, posture invalid으로 종료 전 붕괴했다.

29-control contract의 offline scorer에는 zero forward에도 nonzero logistic quality가 생기는 문제와 support-foot position을 쓰지 않는 문제가 있었고, `*_v1_SUPERSEDED` 원본을 보존한 채 수정했다. 같은 trace를 33 controls로 read-only rescore하면 controls 119–122의 support/lift/forward quality가 잡힌다. 이는 **약 4 control의 phase-window misalignment evidence**지, 33-control intervention의 새 physics/PPO 성공 증거는 아니다.

### R33/R34 side-attribution 오류

64 env×100 control F2 profile/duration sweep은 2초 bounded no-step hard-safety exposure이며 hard event 0, optimizer 0, completed 10 s episode 0, active censoring 64다. `PASS`는 보행 완료가 아니다.

기존 generic `post_swing_*`/`post_stance_share` 분석은 baseline `planned_swing_side=none`과 knee candidate `planned_swing_side=left`를 같은 발처럼 빼는 오류가 있었다. 같은 anatomical foot으로 재계산한 control 91 결과는 다음과 같다.

| 후보 vs lateral-only | 같은 left swing Fz 변화 | 같은 right stance share 변화 |
|---|---:|---:|
| R33 knee10 mrad | −6.4645 N | +.0116975 |
| R33 knee20 mrad | −12.2378 N | +.0232116 |

원래 generic headline (−31.479/−37.252 N, +.0725/+.0840)은 비교 foot이 달라 과장됐다. candidate knee-applied row의 terrain-relative clearance ≥1 mm는 0이고 최대 clearance 약 −.641 mm였다. baseline에는 모든 64 env의 반대편 foot clearance가 없어 same-anatomical geometry delta는 `NOT_COMPARABLE`/`NOT_MEASURED`다. 새 분석은 반드시 explicit left/right raw Fz, side, generation, control, pre/post timestamp로 수행해야 한다.

## 8. Candidate family map: 무엇을 다시 실행하면 안 되는가

아래는 대표 계열의 질문과 현재 결론이다. 각 계열 안의 수많은 재시도/branch는 `docs/results/EXPERIMENT_CATALOG.{csv,jsonl}`의 path와 `docs/methods/EXPERIMENT_FAMILY_INDEX_KO.md`를 따른다.

| family | 주 질문 | 보존된 교훈 | 새 PC에서 재실행 전 필요한 것 |
|---|---|---|---|
| `walk_update1000_safe_direct_*` | SafeDirect로 교대 swing/TD가 가능한가 | Update1700 historical development evidence가 가장 강함 | source/asset/PD/checkpoint exact identity, independent seed evaluation |
| `x2_terrain_*`, `x2_uphill_*`, R25–R53 | ramp/heading/perception adaptation이 flat gait를 보존하는가 | sensor router/late-depth causal claims은 physical parity 미달 또는 U30 regression | terrain geometry, camera owner/freshness, action path, same-cohort frozen comparison |
| `x2_u100_*`, `x2_u105_*` | left/right ankle-roll impact asymmetry의 실제 원인 | contact ledger/timebase boundary가 중요; proxy torque로 원인 확정 금지 | pair identity, actual contact point/wrench/impulse, matched prestate |
| `x2_u230_*`, `x2_u730_*` | reference/phase/upper-body와 course repair | survival/alternation은 가능하나 straightness가 별도 병목 | heading/cross metrics, actor/critic lineage, reference vs physical contact 분리 |
| `x2_lift_*`, `x2_single_support_*`, `x2_pd_*` | 1–10 mm lift와 target authority | low clearance/reward hack/height collapse가 반복 | support transfer→unload first, target-reference corridor audit, native tracking evidence |
| `x2_*teacher*`, `x2_*imitation*` | privileged reference/teacher가 lift를 만들 수 있는가 | action authority c0 lift는 관측됐지만 confirmed TD/height recovery 없음 | teacher frozen qualification before BC/DAgger, no privileged leakage |
| `x2_v9`–`x2_v18` | contact takeover/support-first | latest precise causal setup; phase-window issue and side attribution correction | bilateral geometry logging, F0 A/A, one change at a time, root-height recovery gate |

## 9. 새 컴퓨터에서 우선 해야 할 일

1. **대형 checkpoint를 먼저 다운받으려 하지 말 것.** 이 문서와 source/config/asset manifest를 기준으로 작은 static validation을 먼저 한다.
2. `docs/runtime/ENVIRONMENT_KO.md`, `docs/runtime/PORTING_KO.md`, `docs/methods/TRAINING_AND_RESUME_KO.md`, `docs/results/RESULTS_AND_LIMITATIONS_KO.md`를 읽고 source/asset/runtime matrix를 만든다.
3. working copy를 별도로 만들고 original/archive를 수정하지 않는다. absolute path, symlink, external sibling repository, asset paths를 새 manifest에 기록한다.
4. 먼저 static import/config/observation/action mapping을 검증한다. 15-D action row↔31-D joint mapping, 32 body IDs, control/physics dt, sensor body/pair, terrain-relative sole definition을 dump한다.
5. frozen baseline A/A에서 identical seed/prestate repeatability, target readback, q/qd, left/right Fz, COM, root height/tilt, termination ledger, capture freshness를 확인한다. true CoP/impulse/motor torque가 없으면 그대로 `NOT_MEASURED`로 기록한다.
6. 가장 작은 single-variable intervention만 한다. V18 설계라면 action/PD/reward를 한 번에 바꾸지 말고: F0 baseline A/A → limited lateral support shift → knee authority probe → 33-control reward-window runtime test 순서를 유지한다.
7. PPO는 physical gate 이후에만 5-update canary로 시작한다. long run 조건은 support transfer가 lift보다 먼저 보이고, ≥1 mm clearance와 root-forward nonnegative/height safety가 frozen repeat에서 재현되는 것이다.
8. W&B/dashboard에는 raw left/right Fz, stance share, swing Fz, COM/support-frame, root height/tilt, target–q gap, sole clearance, side, phase, episode generation, qualified/confirmed ledger를 모두 별도로 남긴다. generic stance/swing column만 남기지 않는다.

## 10. metric 용어집

| 용어 | 안전한 의미 | 사용 시 주의 |
|---|---|---|
| `qualified touchdown` | run-specific gate를 통과한 touchdown event | Update1700 parser의 5 mm/force/support 규칙과 최신 ledger는 다를 수 있다. |
| `confirmed balanced landing` | 접촉 후 지속 support/load/timing 조건을 통과한 별도 event | qualified TD와 합치거나 성공 횟수를 공유하지 않는다. |
| `physical lift ≥1 mm` | terrain-relative clearance 조건을 만족한 run-specific foot event | world foot-link z, relative rise, terrain clearance를 구분한다. |
| `sole rise mean` | 그 metric가 집계한 sample population 평균 | conditional physical-lift 평균과 all-control unconditional 평균을 혼용하지 않는다. |
| `stance share` | `stance_Fz/(left_Fz+right_Fz)` 같은 load partition | total Fz/freshness/pair identity가 유효한지 함께 기록한다. |
| `CoP` | contact point+wrench/moment에서 계산한 pressure center | Fz-weighted foot center는 true CoP가 아니다. |
| `PASS` | report가 명시한 scope 통과 | static, launch, bounded safety, frozen gait 모두 다른 PASS다. |
| `lmo_base_height_below` | root/base height threshold 아래에서 episode termination | 낙상·접촉 loss·전진 실패의 원인을 단독으로 설명하지 않는다. |
| PD proxy | software (K_p(q_t-q)-K_d\dot q) | native motor torque/hardware torque 계측이 아니다. |

## 11. 원본 위치와 읽기 우선순위

### 첫 번째로 읽을 이관 문서

1. `x2_locomotion_migration_20261124/docs/methods/METHODS_KO.md`
2. `x2_locomotion_migration_20261124/docs/methods/TRAINING_AND_RESUME_KO.md`
3. `x2_locomotion_migration_20261124/docs/results/RESULTS_AND_LIMITATIONS_KO.md`
4. `x2_locomotion_migration_20261124/docs/results/LATEST_R33_R34_REVIEW.json`
5. `x2_locomotion_migration_20261124/docs/runtime/ENVIRONMENT_KO.md`와 `PORTING_KO.md`

### 상세 원본 근거

- historical qualified gait: `walk_update1000_safe_direct_envlocal_100env_to3000_20261029/SWING_TOUCHDOWN_VALIDATION_REPORT.md`
- U730 course/survival separation: `x2_u230_course_scope_repair_u730_analysis_20261109/U730_FINAL_RESULT.md`
- R53 small PPO regression: `x2_terrain_phase_heading_r53_ppo_resume/R53_TRAINING_REPORT.md`
- U17 support transfer funnel: `x2_u17_weight_transfer_closure_20261120/FINAL_REPORT.md`
- privileged Teacher/Student status: `x2_privileged_walking_teacher_student_20261117/FINAL_REPORT.md`
- imitation authority U5: `x2_safe_landing_teacher_il_u20_v8_reference_hold_measurement_recovery_20261122/FINAL_REPORT.md`
- V18 support-first formulation and raw outcome: `x2_v18_support_first_method_redesign_20261123/METHOD_REDESIGN.md`, `FINAL_REPORT.md`, `final_verdict.json`
- R33/R34 corrected interpretation: `x2_locomotion_migration_20261124/docs/results/LATEST_R33_R34_REVIEW.json`
- all named experiment paths/results: `x2_locomotion_migration_20261124/docs/results/EXPERIMENT_CATALOG.csv` and `.jsonl`
- all method-family paths: `x2_locomotion_migration_20261124/docs/methods/EXPERIMENT_FAMILY_INDEX_KO.md`

## 12. Explicit limitations handed to the next team

- The archive is a point-in-time file preservation, not a complete simulator-state snapshot.
- 3,448 historic checkpoint snapshots totaling 18,061,367,099 bytes were already deleted before migration; documents may mention them but payloads cannot be recovered from this archive.
- Original `/data` targets and some symlinks are unavailable; do not silently replace them with a different asset.
- Several reports are inherited/copied across branches. A `FINAL_REPORT.md` can describe an older U5 result; use per-branch manifest/result/checkpoint counter/source pin for current state.
- R33/R34 generic geometric foot comparisons are `NOT_COMPARABLE`; complete 64-env same-anatomical-foot clearance evidence is `NOT_MEASURED`.
- true CoP, contact/friction impulse and native motor torque are absent in latest V18 capture. No document may infer them from proxies.
- New-PC native execution/physics/PPO qualification remains `NOT_RUN` until performed under the new runtime and asset hashes.

## 13. Bottom-line research state

The archive contains historical evidence that X2-like cyclic touchdown can be achieved in a particular SafeDirect development cohort, and later evidence that course control, ramp transfer, landing recovery, and realistic support transfer remain unresolved in other branches. The current defensible bottleneck is not simply ``increase lift reward`` or ``increase knee stiffness``. It is the causal chain:

\[
\text{validated contact/load state}
\rightarrow \text{COM and stance-load transfer}
\rightarrow \text{swing-foot unloading}
\rightarrow \text{terrain-relative lift}
\rightarrow \text{forward swing}
\rightarrow \text{touchdown}
\rightarrow \text{new-foot load acceptance and height recovery}.
\]

The next work should preserve this ordering, use same-foot left/right measurements, and promote models only when the physical gates and independent frozen evaluation agree. Neither high reward, low KL, finite PPO loss, a bounded safety PASS, nor a visually plausible Viser trace alone is sufficient.

