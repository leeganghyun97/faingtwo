# 독립 누락·정합성·공개 범위 검토

검토일: 2026-11-24 KST  
검토 범위: 현재 소형 handoff 초안, 세 agent 초안, 그리고 기준 이관본
`x2_locomotion_migration_20261124`의 methods/runtime/results/portability 문서.  
검토 방식: 원본 재실행·Isaac·PPO·checkpoint 역직렬화 없이 문서와 경로·계약만 대조했다.

## 판정

**문서 패키지는 연구 이력과 현재 병목을 상당히 잘 보존한다.** 특히 `raw action → target → q/qd → foot/contact`, reward-hacking, CoP/torque 권한 한계, CReF와 후속 HDMI 계열의 분리는 적절하다.

다만 아래 **P0 항목을 반영하고 실제 clone 검증을 끝내기 전에는 "새 PC에서 바로 재현 가능" 또는 "공개 다운로드 준비 완료"라고 말하면 안 된다.** 이 패키지는 대용량 원본의 대체 실행물이 아니라, 문서 기반의 안전한 의사결정·추가 자료 요청용 인계본이어야 한다.

## P0 — 공개/다운로드 전에 반드시 수정 또는 검증

| ID | 발견 내용 | 왜 중요한가 | 필요한 수정·검증 |
|---|---|---|---|
| P0-1 | [00_CURRENT_STATE_AND_DECISION.md](../00_CURRENT_STATE_AND_DECISION.md)의 “연속 보행은 아직 검증되지 않았다”는 표현이 [01 방법론 초안](01_METHODOLOGY_AND_EXPERIMENT_HISTORY.md)의 Update1700 development-cohort 20-env × 10 s touchdown/alternation 기록과 겉으로 충돌한다. | 역사적 강한 evidence를 지우거나, 반대로 현 CReF/V18의 미검증을 성공으로 오해하게 한다. | 다음처럼 범위를 분리한다: **“현재 CReF/Teacher/V18의 새 runtime에서 독립적으로 검증된 연속 보행은 없다. 다만 SafeDirect development cohort의 Update1700에는 제한된 20-env·10 s historical gait evidence가 있다.”** Update1700, U730, CReF-2k, U17, V18을 같은 policy/qualification으로 취급하지 않는 표를 추가한다. |
| P0-2 | observation 차원이 서로 달라 보인다. 01 초안은 2026-11-17 Teacher를 `94 + 824 + 15 = 933D`로, [02 런타임 초안 197행](02_RUNTIME_REPRODUCTION_AND_COMMANDS.md#L197)은 “V18 teacher privileged actor”를 `94 + 824 = 918D`로 적는다. | checkpoint와 loader를 잘못 묶으면 partial load·관측 누수·비재현이 생긴다. | 각각의 **source package/run/actor observation/critic observation/action dimension/checkpoint format**을 한 행씩 갖는 `LINEAGE_COMPATIBILITY_MATRIX`를 만든다. 918D가 V18의 별도 actor인지, 933D가 15D reference를 포함한 11/17 Teacher인지 source pin으로 확정하고 용어를 고친다. |
| P0-3 | 상위 README와 다운로드 문서는 `VERIFY_HANDOFF.py`, `HANDOFF_MANIFEST.json`, public branch를 이미 전제로 하지만, 이 검토 시점에는 해당 파일과 실제 branch clone PASS가 아직 확인되지 않았다. | 사용자는 대용량 대신 이 명령으로 내려받아야 한다. 존재하지 않는 verifier/branch는 handoff 실패다. | publish 직전에 manifest에 **모든 공개 파일의 상대경로·SHA-256·바이트 수·생성 시각·source commit**을 넣고 verifier를 실행한다. 그 다음 깨끗한 임시 디렉터리에서 `git clone --depth 1 --branch ...` → `python3 VERIFY_HANDOFF.py`를 실제로 통과시킨 receipt를 남긴다. |
| P0-4 | 문서-only 다운로드와 full archive 복원이 여러 문서에 설명되지만, 첫 화면에서의 구분이 충분히 강하지 않다. | 새 컴퓨터에서 문서만 받고 Isaac/PPO를 실행하려는 오류를 막아야 한다. | README 첫 화면에 2열 표를 둔다: **문서 handoff: 작은 public download, 실행 불가** / **full archive: private + 대용량 + source/asset/runtime 재연결 필요, 그래도 native qualification 미완료**. `04_NEW_PC_CODEX_PROMPT_KO.md`에도 “문서만 있으면 실행 명령을 제안하지 말고 inventory와 missing-input request만 작성”을 굵게 넣는다. |
| P0-5 | 공개 `faingtwo` branch와 private full archive의 접근 범위가 혼동될 여지가 있다. | `faingtwo`의 unrelated main 프로젝트를 X2 source로 오인하거나 private repo 404를 archive 소실로 해석할 수 있다. | 다운로드 문서에 “public branch에는 **이 문서 폴더만** 있고 X2 source/checkpoint/asset은 없다. private archive 404는 인증 부재도 가능하다”를 한 문장으로 고정한다. public repo의 `main`은 건드리지 않고 branch/하위 docs 경로만 사용한다. |
| P0-6 | 런타임 초안에는 실제 native frozen/PPO 명령이 포함된다. 주석은 좋지만, 문서 묶음만으로 실행 가능한 것처럼 복사될 위험이 있다. | 원본 asset/PD/guard/one-shot 범위 없이 실행하면 실험 identity와 안전 계약이 깨진다. | Section 12–13 명령의 바로 앞에 **“reference-only; full artifact set, immutable working copy, static gate, native authorization이 없으면 실행 금지”**를 넣는다. `F0/F1`의 measurement authority가 통과하기 전 PPO 명령을 실행하지 않는 결정표를 top-level에도 링크한다. |

## P1 — 정확성·재현성·누락 방지에 필요한 보완

| ID | 발견 내용 | 필요한 보완 |
|---|---|---|
| P1-1 | 기준 archive Git commit은 `0dee161…`이지만 검토 당시 working tree에는 `drive_tools/CLI_HANDOFF.json`, `drive_tools/REMOTE_INDEX_ERROR.json`의 미커밋 변경이 있었다. | `SOURCE_BASIS.md` 또는 manifest에 commit, 검토 일시, 두 dirty path와 역할을 기록한다. 과학적 X2 source/result 근거에 포함하지 않았는지, 또는 포함했다면 checksum을 명시한다. |
| P1-2 | full archive는 이미 삭제된 checkpoint 3,448개(18,061,367,099 bytes), `/data` 의존성, 45 broken symlink, 698 unresolved URDF/MJCF reference를 가진다. | top-level “복원 불가/외부에서 받아야 하는 입력” 표에 포함한다. full archive도 strict replay를 보장하지 않으며, 모델 파일명만으로 exact continuation을 만들 수 없음을 강조한다. |
| P1-3 | 대용량을 피하려는 사용자의 목적에 비해 full-recovery disk budget이 명시적이지 않다. | docs package 크기와 별도로 **full snapshot logical 171,059,417,442 bytes**, transfer payload 약 109.9 GB, working copy는 filesystem reflink가 없으면 추가 대용량 공간이 필요하다고 적는다. 충분한 여유 공간 없이 full restore를 시작하지 않는 명령을 넣는다. |
| P1-4 | evidence가 풍부하지만, direct measurement / static check / derived offline rescore / hypothesis의 범주가 문서마다 분산돼 있다. | 핵심 결론마다 source path, source kind, physical execution 여부, side/timebase comparability, 권한 상태를 가진 작은 `EVIDENCE_STATUS_TABLE`을 추가한다. 특히 U17 funnel, CReF 2k collapse, Update1700, U730, V18 R33/R34를 한 표에서 구분한다. |
| P1-5 | `COM margin > +10 mm`와 3-control 지속, 1/5 mm gate 등의 수치가 future gate인지 historical universally validated threshold인지 혼동될 수 있다. | 모든 이런 값에 **“제안된 next-experiment gate; 제조사/보편 임계값 아님”** 또는 원래 run-specific definition을 붙인다. U1700 touchdown parser와 later qualified/confirmed/physical-lift 판정기가 다르다는 점을 같은 표에 둔다. |
| P1-6 | 03 초안의 `COM_TRANSFER_AND_FOOT_PLACEMENT`는 가장 강한 CReF matched replay 결론이지만 모든 X2 branch의 단일 확정 원인처럼 읽힐 수 있다. | “해당 frozen CReF matched-state analysis에서 지지된 primary mechanism”으로 한정한다. U17/V18에서 sensor pair/CoP/native effort의 authority가 다르거나 미측정인 부분은 별도 일반화 금지 문장을 둔다. |
| P1-7 | strict resume/warm start 설명은 좋지만 `--resume`/`--warm-start` mutual exclusion과 dimension/normalizer/RNG semantics는 runner-specific이다. | compatibility matrix에 loader 이름과 runner scope를 붙이고, 새 host는 tensor keys/shapes·normalizer·optimizer·recurrent sidecar를 **print-only audit**한 뒤 mismatch를 reconstruction으로 분류하게 한다. |
| P1-8 | source map은 충분하나 top-level read order에 `07_ARTIFACT_REQUEST_TEMPLATE_KO.md`와 이 검토 파일이 빠져 있다. | README의 읽기 목록에 추가한다. 문서 only 사용자는 07의 “작은 추가 자료 요청 5묶음”으로 끝나게 하며, full archive 다운로드를 기본 경로로 만들지 않는다. |
| P1-9 | asset/license scope를 공개 문서에서 더 명확히 해야 한다. Isaac, Agibot asset, ACCAD/SMPL-X/GMR, external code의 redistribution 권한은 이 handoff가 부여하지 않는다. | README 공개 제한에 “경로·hash·메타데이터의 기록은 redistributable license 또는 실행 권한을 의미하지 않음”을 추가한다. public branch에 raw checkpoint, URDF/USD/mesh, video, W&B export, Drive signed URL을 넣지 않는 final scan을 수행한다. |

## P2 — 사용성과 유지보수 개선

| ID | 개선 | 이유 |
|---|---|---|
| P2-1 | `python`과 `$PYTHON`을 섞은 archive validation 예시는 어느 interpreter가 필요한지 한 줄 설명한다. | `check_restored.py`가 stdlib인지, Isaac prefix를 요구하는지를 혼동하지 않게 한다. |
| P2-2 | 각 실행 명령 옆에 “정적 검사 / native physics / PPO” 배지를 붙인다. | `import`, `one physics step`, frozen evaluation, optimizer 0, PPO update를 사용자가 명확히 구분하게 한다. |
| P2-3 | 전체 한국어 문서의 끝에 `GLOSSARY`와 약어 표를 둔다. | CReF, F0/F1/F2, Q_S/Q_L/Q_F, SafeDirect, strict resume, warm start, qualified/confirmed TD, physical lift의 다의성을 줄인다. |
| P2-4 | 새 PC가 남겨야 할 네 개의 첫 산출물(`NEW_HOST_INVENTORY`, `MISSING_INPUTS`, `F0_F1_CAPTURE_CONTRACT`, `FIRST_AAB_PLAN`)을 파일명·필수 열·PASS 조건으로 템플릿화한다. | Codex가 바로 PPO로 넘어가는 대신 재현 가능하고 검토 가능한 출발점을 만든다. |

## 공개 자료·민감 정보 점검

현재 handoff 초안에서 직접적인 API key, bearer token, `.netrc` credential, W&B key, Drive signed download URL은 발견하지 못했다. `leeganghyun97` GitHub namespace와 `/home/fain`, `/data/fain-data` 같은 과거 경로는 의도적으로 문서화돼 있으나 **새 PC의 경로로 복사해 사용할 값은 아니다.**

공개하기 전 다음 최종 검사를 수행해야 한다.

1. `rg -n -i '(api[_ -]?key|token|password|secret|authorization:|bearer|netrc|cookie)'`로 공개 대상 파일만 검사한다.
2. 대상 tree가 문서·manifest·verifier만 포함하고 `.pt/.onnx/.usd/.urdf/.zip/.tar/.csv` 대형 trace, `wandb/`, cache, credential 파일을 포함하지 않는지 검사한다.
3. public upload는 source code/asset 라이선스나 개인 Drive 공유 권한을 바꾸지 않는다고 README에 명시한다.
4. generated manifest의 SHA와 branch clone 결과를 독립 위치에서 대조한다.

## 최종 권고 순서

1. **P0-1~P0-6을 반영한다.**
2. public 대상 파일의 secret·binary·license scope scan을 수행한다.
3. `HANDOFF_MANIFEST.json`과 `VERIFY_HANDOFF.py`를 생성하고 local PASS를 얻는다.
4. `faingtwo`의 새 branch/새 docs path에만 올린다.
5. 깨끗한 디렉터리에서 public clone과 verifier를 재실행해 download receipt를 남긴다.
6. 그 뒤에만 “문서 인계 다운로드 가능”이라고 안내한다. full archive/native reproduction은 별도 private access·storage·runtime qualification 작업으로 남긴다.

## 이 검토에서 확인한 가장 중요한 한 줄

**새 컴퓨터가 먼저 복원해야 하는 것은 reward 숫자가 아니라 lineage·asset·sensor/timebase·action authority의 증거 사슬이며, 문서-only handoff는 그 사슬을 설계·감사하는 데 충분하지만 원래 PhysX 실험을 실행하는 데는 충분하지 않다.**
