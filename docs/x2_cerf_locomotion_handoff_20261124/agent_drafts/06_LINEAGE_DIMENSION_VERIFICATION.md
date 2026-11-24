# CReF/X2 관측 차원·체크포인트 계보 검증

**감사 범위:** 2026-11-24의 로컬 정적 소스·계약·체크포인트 텐서 감사만 수행했다. Isaac 물리 실행, PPO, optimizer, checkpoint load/save, 원본 소스 변경은 수행하지 않았다.

## 결론

아래 숫자는 같은 정책 입력 차원이 아니다. 특히 `918D`는 두 Teacher 계열에서 **native physics 증강 벡터가 expert context를 붙이기 전의 중간 차원**이다. 검증된 전체 Teacher actor 차원은 2026-11-17 계열에서 `933D`, V18 소스/계약에서는 `948D`다. 따라서 `918D V18 privileged actor`라고 부르거나 `933D`와 `948D` 체크포인트를 같은 형식으로 resume하면 안 된다.

V18 P0 U5 checkpoint는 더 중요한 정적 불일치를 보인다. 저장 계약은 외부 actor observation `948D` 및 critic `811D`라고 기록하지만, checkpoint 내부 base-network 텐서는 `87D` base 입력을 가리킨다. 이 차이는 `temporal_support_residual.py`의 87D core/94D outer wrapper 설계와 관련 있을 수 있으나, 현재 보존된 소스만으로 실제 P0 실행 import chain과 strict load 호환성이 증명되지는 않았다. P0 checkpoint를 다른 Teacher 구현에 복원하기 전에 정적 호환성 검증이 필요하다.

## 사실 행렬

| 계열 / 식별자 | actor 관측 계약 | critic 계약 | action | 체크포인트·증거 상태 | 판정 |
|---|---:|---:|---:|---|---|
| **Native HDMI MLP baseline** (`X2/cref2603/teacher`) | `803D` native policy observation | `803 + 809 = 1612D` | 15D | `outputs/isaac_cref_15000_terrain/checkpoints/checkpoint_step_000015000.pt`가 기준 문서에 지정됨 | CReF recurrent adapter가 아닌 native MLP baseline. 803/809/1612는 이 계열에만 사용한다. |
| **CReF recurrent adapter / static closure** | `85D` proprio + optical depth `[1,48,64]` | `85 + 3 + 714 = 802D` | 15D | `outputs/isaac_cref_static_closure_2k_20261002/checkpoints/checkpoint_000007.pt`, SHA-256 `4dc2de7ea782074f619cb9bb9601c11e5567a541c35de19aaa5f130294757d60` | 803D native MLP과 다른 adapter contract. 이 checkpoint의 텐서가 85/802를 직접 뒷받침한다. |
| **2026-11-17 privileged Teacher** | base `94D` + native physics `824D` + expert context `15D` = **`933D`** | base critic은 소스상 `94 - 7 + 3 + 714`, 별도 teacher branch `824+15` | 15D | `x2_privileged_walking_teacher_student_20261117`의 source 및 `TEACHER_MODEL_CONTRACT.json`/`OBSERVATION_AUTHORITY.json` | `918D=94+824`는 expert context 추가 전 중간 값. 저장된 실행 checkpoint의 존재는 이 감사에서 확인되지 않았다. |
| **V18 privileged Teacher source / P0 contract** | base `94D` + native physics `824D` + expert ref/action context `30D` = **`948D`** | contract상 `811D` | 15D | `x2_v18_support_first_method_redesign_20261123/frozen_runtime_source`; P0 `walk_observation_action_contract.json` | 소스와 saved contract는 V18 actor를 948D로 명시한다. `918D`는 여기서도 중간 physics vector일 뿐이다. |
| **V18 P0 U5 checkpoint 내부 텐서** | outer 948D input 경로는 contract에만 기록됨; 내부 base core는 텐서상 `87D` | 텐서상 `87 + 3 + 714 = 804D` | 15D | `ppo_runs/P0_reward_window_33_u5_r3/checkpoints/walk_update_0005.pt`, SHA-256 `555316261eb9511616ab4c27b12adf0476c33385e697bc38fb3dad9ac2eb2b95` | source/contract와 checkpoint의 base-core 차원을 하나의 검증되지 않은 단일 모델로 취급하면 안 된다. 정적 strict-load 검증 전 resume 금지. |

## 차원 계산과 원본 증거

### 1. 803D / 809D / 1612D: native MLP baseline

`/home/fain/Desktop/RoboJuDo-Plus/reports/cref_recurrent_isaac/MLP_PPO_BASELINE_FINGERPRINT.md:3-24`는 다음을 명시한다.

> “This baseline is **not the paper CReF implementation**.”
>
> “Actor | `MLP803→512→256→256→15`”
>
> “Critic | `concatenated803+809=1612 input → scalar value`”

같은 문서의 11-12행은 native `PPOPolicy`와 GRU/LSTM 부재를 적고 있다. 따라서 이 숫자를 CReF recurrent actor의 input shape로 부르면 안 된다. `CREF_ISAAC_RUNTIME_CONTRACT.md:25`도 “native MLP observation remains 803 and native privileged group remains 809”이라고 분리한다.

### 2. 85D + depth / 802D: CReF recurrent adapter static closure

`/home/fain/Desktop/HDMI/outputs/isaac_cref_static_closure_2k_20261002/contract.json:2-16`의 saved runtime contract는 다음을 기록한다.

> `"actor_proprio_dim": 85`
>
> `"actor_depth_shape": [1, 48, 64]`
>
> `"critic_input_dim": 802`
>
> `"critic_terrain_dim": 714`, `"action_dim": 15`

feature slice는 같은 파일 17-53행에서 `4 + 3 + 3 + 15 + 15 + 45 = 85`로 확인된다. `/home/fain/Desktop/RoboJuDo-Plus/reports/cref_recurrent_isaac/CREF_ISAAC_RUNTIME_CONTRACT.md:27-40`도 85D 구성과 다음 수식을 명시한다.

> “Critic input is 85 proprio + 3 GT velocity + 714 native terrain heights = 802.”

checkpoint를 CPU에서 읽은 정적 state-dict shape도 일치한다.

| key | shape | 직접 확인되는 관계 |
|---|---:|---|
| `proprio_tokenizer.0.weight` | `(64, 88)` | `85 + 3` |
| `velocity_lstm.weight_ih` | `(256, 149)` | `85 + 64` |
| `critic.0.weight` | `(256, 802)` | critic input 802 |
| `action_head.2.weight` | `(15, 64)` | 15D action |

이는 native 803/809 MLP checkpoint와 tensor shape가 호환되지 않는다는 직접 증거다.

### 3. 2026-11-17 privileged Teacher: 918D physical vector, 933D actor

`/home/fain/Desktop/HDMI/x2_privileged_walking_teacher_student_20261117/teacher_observation.py:29-50`은 `BASE_PROPRIO_DIM = 94`와 16개 native physics feature width의 합인 `PRIVILEGED_DIM`을 정의한다. listed widths의 합은 `824`다. 따라서 이 모듈의 `TEACHER_PROPRIO_DIM`은 `94 + 824 = 918`이다.

그러나 actor의 최종 입력은 별도 expert context를 붙인 후의 값이다. `/home/fain/Desktop/HDMI/x2_privileged_walking_teacher_student_20261117/teacher_expert_reference.py:15-21`은 다음과 같다.

> `EXPERT_CONTEXT_DIM = 15`
>
> `TEACHER_PROPRIO_DIM = BASE_PROPRIO_DIM + PRIVILEGED_DIM + EXPERT_CONTEXT_DIM`

그러므로 실행 actor contract는 `94 + 824 + 15 = 933D`다. `/home/fain/Desktop/HDMI/x2_privileged_walking_teacher_student_20261117/teacher_policy.py:23-29,69-73`은 933D/15D/714D 검사를 수행하고 `actor_proprio_dim`을 이 최종 `TEACHER_PROPRIO_DIM`으로 반환한다. 이 경로의 `OBSERVATION_AUTHORITY.json` 및 run별 `TEACHER_MODEL_CONTRACT.json`도 actor `933D`, physical columns `[94,918)`, expert-reference columns `[918,933)`로 기록한다.

**문서 충돌:** `DISTILLATION_IMPLEMENTATION.md`의 “918D frozen privileged Teacher observation” 서술은 source/contract 기준 full actor input과 맞지 않는다. 그것은 expert append 전 physical observation을 뜻했을 가능성이 높지만, 문서만으로는 확정할 수 없다. 이 감사에서 `runs/` 아래의 actual `.pt` Teacher checkpoint는 찾지 못했으므로, 해당 prose만으로 trained artifact identity를 주장할 수 없다.

### 4. V18: 918D physical vector, source/contract상 948D actor

V18 frozen source의 `/home/fain/Desktop/HDMI/x2_v18_support_first_method_redesign_20261123/frozen_runtime_source/teacher_observation.py:29-50`은 2026-11-17과 같은 `94D + 824D = 918D` physical vector를 정의한다.

그러나 `/home/fain/Desktop/HDMI/x2_v18_support_first_method_redesign_20261123/frozen_runtime_source/teacher_expert_reference.py:38-42`은 다음을 명시한다.

> `EXPERT_REFERENCE_FEATURE_DIM = 15`
>
> `EXPERT_ACTION_TARGET_DIM = 15`
>
> `EXPERT_CONTEXT_DIM = EXPERT_REFERENCE_FEATURE_DIM + EXPERT_ACTION_TARGET_DIM`
>
> `TEACHER_PROPRIO_DIM = BASE_PROPRIO_DIM + PRIVILEGED_DIM + EXPERT_CONTEXT_DIM`

즉 V18 source 기준 full actor는 `94 + 824 + (15+15) = 948D`다. P0 run의 `/home/fain/Desktop/HDMI/x2_v18_support_first_method_redesign_20261123/ppo_runs/P0_reward_window_33_u5_r3/walk_observation_action_contract.json:83-84`도 `"actor_proprio_dim": 948`이라고 기록한다. 따라서 “V18 918D privileged actor”라는 명칭은 source 및 saved contract와 모순된다.

### 5. V18 P0 checkpoint의 base-core 불일치

P0 U5 checkpoint에서 정적으로 읽은 tensor는 아래와 같다.

| key | shape | 산술적 의미 |
|---|---:|---|
| `proprio_tokenizer.0.weight` | `(64, 90)` | `87 + 3`, 94+3이 아님 |
| `velocity_lstm.weight_ih` | `(256, 151)` | `87 + 64`, 94+64가 아님 |
| `critic.0.weight` | `(256, 804)` | `87 + 3 + 714`, 811이 아님 |
| `teacher_encoder.0.weight` | `(128, 854)` | `824 + 30`; V18 teacher context는 일치 |
| `action_head.2.weight` | `(15, 64)` | 15D action은 일치 |

이러한 87D base shape는 `/home/fain/Desktop/HDMI/x2_v18_support_first_method_redesign_20261123/frozen_runtime_source/temporal_support_residual.py:12-13`의 `BASE_PROPRIO_DIM = 87`과 부합한다. 같은 파일 172-176행은 outer temporal-support observation을 `87 + 2 + 2 + 3 = 94`로 정의하고, 356-361행은 `94D` outer 입력을 검사한 뒤 `87D` base policy로 `super().__init__`한다.

이는 **설명 후보**일 뿐, P0가 실제로 이 wrapper/class/import chain으로 실행됐다는 증명은 아니다. 반대로 V18 `teacher_policy.py:48-54`는 copied Teacher source에서 `BASE_PROPRIO_DIM`을 `teacher_observation.py`의 94로 import하여 base policy를 만들도록 보인다. 저장된 P0 contract (`948/811`), copied Teacher source (`94D core`로 읽힘), checkpoint tensor (`87D core`)의 세 증거는 현 시점에 하나의 strict-load-compatible model identity로 닫히지 않는다.

## 호환성 규칙

| 출발 artifact | 대상 구현 | 결론 |
|---|---|---|
| 803/809 native MLP checkpoint | 85D CReF recurrent adapter | **비호환.** actor/critic architecture와 observation contract가 다르다. |
| 85D/802 static-closure checkpoint | 803/809 native MLP | **비호환.** recurrent/depth estimator와 critic shape가 다르다. |
| 933D 2026-11-17 Teacher | V18 948D Teacher | **직접 resume 불가.** 15D expert context와 30D expert-reference/action context가 다르다. |
| P0 U5 checkpoint | copied V18 teacher source의 직접 strict load | **미검증·보류.** base core 87/94와 critic 804/811 불일치를 먼저 해소해야 한다. |
| P0 U5 checkpoint | P0의 정확한 original wrapper/runtime | **가능성은 있으나 아직 증명되지 않음.** original launch import graph와 strict-load receipt가 필요하다. |

## 확정되지 않은 항목과 필요한 최소 검증

1. **P0 exact runtime identity:** `MODULE_ORIGIN_RECEIPT.json`/source integrity 기록은 존재하지만, 실제 process의 `type(model)`, `__module__`, outer `input_proprio_dim`, inherited base dimensions, `sys.path`가 저장되어 있지 않다.
2. **P0 strict load:** 위 checkpoint를 P0 original launch path에서 `strict=True`로 load했는지, 어느 missing/unexpected key도 없었는지의 receipt가 없다.
3. **2026-11-17 saved Teacher artifact:** source/contract는 933D를 분명히 하지만, 이 감사 범위의 `runs/`에서 해당 Teacher `.pt`를 찾지 못했다. model source와 run result를 trained checkpoint로 혼동하면 안 된다.
4. **918D prose의 의미:** 2026-11-17 문서의 918D 표현은 source와 run contract의 full actor 933D와 불일치한다. expert context 이전 vector를 뜻하는지 작성 당시의 구형 contract인지는 문서만으로 확정할 수 없다.

P0를 재개하기 전 권장되는 유일한 최소 검증은 **native physics 및 optimizer 없이** P0의 original runner import path로 model을 instantiate하고 다음을 JSON receipt로 남기는 것이다: `(a)` model/wrapper classes와 module paths, `(b)` outer actor input dimension, `(c)` inherited base input dimension, `(d)` tokenizer/LSTM/critic/teacher encoder shapes, `(e)` checkpoint `strict=True` load 결과, `(f)` 948D outer observation이 87D core와 824D+30D teacher branch로 slice되는 정확한 코드 경로. 이 receipt가 생길 때까지 P0 U5를 933D Teacher 또는 독립 948D copied source에 resume하지 않는다.

## 감사 제한

- 이 문서는 성공적인 보행, native runtime pass, checkpoint execution을 주장하지 않는다.
- hash와 tensor shape는 로컬 파일의 읽기 전용 검사 결과다.
- 어느 reward, PD, action routing, Guard, asset, checkpoint, source file도 이 감사에서 변경되지 않았다.
