# X2 CReF Locomotion — 소형 인계 패키지

작성일: 2026-11-24 (KST)  
대상: 다른 컴퓨터에서 X2 CReF/Teacher 기반 보행 연구를 **논리와 증거를 잃지 않고** 이어갈 사람 또는 Codex

## 먼저 구분할 두 가지

| 받는 것 | 크기·접근 | 할 수 있는 일 | 할 수 없는 일 |
|---|---|---|---|
| **이 문서 handoff** | 작음, public Git branch | 연구 이력 검토, 다음 실험 설계, 필요한 소형 artifact 요청 | Isaac/PPO 실행, checkpoint 재생, original gait 재현 |
| **full X2 archive** | private, 대용량, runtime/asset 재연결 필요 | 원본 source·artifact의 복원 작업 | 자동적인 strict replay 또는 native gait qualification 보장 |

`faingtwo`의 `main`은 별도 Genie 프로젝트다. X2 자료는 이 문서용 branch의 아래 `docs/` 폴더에만 있으며, X2 source/checkpoint/asset은 포함하지 않는다.

## 이 패키지의 목적

원래 X2 이관 아카이브는 체크포인트, 원시 trace, Isaac 자산, 중복 런타임을 포함해 내려받기에는 지나치게 크다. 이 폴더는 그 대용량 파일을 복제하지 않고 다음을 전달한다.

- 어떤 소스 체계와 관측·제어 계약을 사용했는지
- 어떤 실험 계열을 시도했고, 무엇이 측정됐거나 측정되지 않았는지
- 실제 보행 성공으로 오인하면 안 되는 결과와 실패 병목
- 다른 컴퓨터에서 분석·복구·다음 실험을 시작하는 안전한 순서
- 필요한 최소 원본 파일을 나중에 선택적으로 요청하는 방법

이 패키지 자체로는 Isaac Sim, URDF/USD/mesh, 체크포인트, W&B 실행 이력, 원시 CSV 또는 모델 바이너리를 제공하지 않는다. 따라서 **동일 체크포인트를 즉시 재생하거나 원래의 물리 실험을 완전히 재현할 수는 없다.** 문서 기반으로 연구 방법과 다음 진단을 재개하는 용도다.

## 반드시 먼저 읽을 파일

1. `00_CURRENT_STATE_AND_DECISION.md` — 현재 결론과 다음 실험의 우선순위
2. `agent_drafts/01_METHODOLOGY_AND_EXPERIMENT_HISTORY.md` — 방법론·실험 계보
3. `agent_drafts/02_RUNTIME_REPRODUCTION_AND_COMMANDS.md` — 환경·실행·재개 계약
4. `agent_drafts/03_DIAGNOSTICS_FAILURES_AND_ARTIFACT_MAP.md` — 물리 병목·진단 근거
5. `04_NEW_PC_CODEX_PROMPT_KO.md` — 새 컴퓨터의 Codex에 붙여 넣을 시작 프롬프트
6. `05_DOWNLOAD_AND_RECONSTRUCTION_COMMANDS_KO.md` — 다운로드와 검증 명령
7. `06_METRICS_AND_EVIDENCE_RULES_KO.md` — 수치 해석 규칙과 금지된 과장
8. `07_ARTIFACT_REQUEST_TEMPLATE_KO.md` — 필요한 작은 원본 자료를 요청하는 양식
9. `08_EVIDENCE_STATUS_TABLE_KO.md` — 서로 다른 success/failure evidence의 범위 비교
10. `agent_drafts/04_INDEPENDENT_GAP_AND_SAFETY_REVIEW.md` — 공개 전 독립 누락·안전 검토
11. `agent_drafts/05_NEW_HOST_EXPERIMENT_PLAYBOOK.md` — 새 호스트의 단계별 실행·판정 playbook
12. `agent_drafts/06_LINEAGE_DIMENSION_VERIFICATION.md` — observation/checkpoint 차원과 strict-resume 금지 조합
13. `PUBLIC_DOWNLOAD_RECEIPT.md` — 공개 branch의 fresh clone·manifest 검증 기록

## 범위와 공개 제한

이 문서 패키지에는 다음을 넣지 않는다.

- checkpoint/ONNX/optimizer state/대규모 rollout·video·Viser trace
- URDF/USD/mesh, Isaac 설치 파일, NVIDIA/Conda 캐시, 외부 상용·제한 라이선스 자산
- API key, W&B/Drive/GitHub 인증 정보, 개인 경로의 비밀값
- 원시 contact/force 데이터가 없는 값을 추정해 만든 접촉 impulse·native torque

원본 증거가 없는 항목은 `NOT_MEASURED`, 실행하지 않은 것은 `NOT_RUN`, 비교 설계가 맞지 않는 것은 `NOT_COMPARABLE`로 기록한다. 이 셋은 실패를 숨기는 값이 아니라 다음 작업의 설계 제약이다. 자산·경로·hash의 문서화는 해당 자산의 재배포 라이선스 또는 실행 권한을 뜻하지 않는다.

## 원본 대형 이관본과의 관계

이 문서의 근거는 로컬 스냅샷 `x2_locomotion_migration_20261124`의 Git commit `0dee161c976e29f4d9caa5770cb1d261c617ed8b` 및 각 실험 산출물에서 정리했다. 원본에는 역사적으로 사라진 checkpoint와 외부 의존성이 남아 있어도 모두 복구된 것은 아니다. 문서의 `source_maps/ORIGINAL_EVIDENCE_MAP.md`가 원본 상대 경로와 필요한 추가 자료를 설명한다.

## 성공 여부에 대한 짧은 결론

**현재 CReF/Teacher/V18 runtime에서 새 환경 독립 평가로 검증된 연속 보행은 없다.** 다만 SafeDirect development cohort의 Update1700에는 20 env × 10 s 범위에서 qualified touchdown/strict alternation을 기록한 역사적 gait evidence가 있다. Update1700, U730, CReF-2k, U17, V18은 policy·asset·parser·qualification scope가 달라 같은 성공률로 취급할 수 없다. 반복해서 확인된 현재 병목은 지지 발으로의 실제 하중 이전, 스윙 발 언로딩, terrain-relative clearance, touchdown 뒤의 지지 인수와 root-height 회복이 하나의 시간축에서 확인되지 않았다는 점이다. reward 증가, reference 추종, 큰 lift 평균, safety PASS, 또는 Viser의 자세가 그 자체로 보행 성공을 뜻하지 않는다.

## full archive를 받으려면 필요한 공간과 한계

기준 archive의 logical snapshot은 약 **171,059,417,442 bytes**, 전송용 compressed payload는 약 **109.9 GB**였다. reflink가 없는 filesystem에서는 restore/working copy에 추가 대용량 공간이 필요하다. 역사적 checkpoint 3,448개(약 18.06 GB), 일부 `/data` 의존성, 45 broken symlink 및 698 unresolved URDF/MJCF reference가 완전 복구 대상이 아니었다. 그러므로 private full archive를 받아도 strict replay가 자동으로 보장되지는 않는다.

## 빠른 다운로드

공개 GitHub 브랜치에서 이 문서만 받으려면 아래 명령을 사용한다. 정확한 주소와 검증 방법은 `05_DOWNLOAD_AND_RECONSTRUCTION_COMMANDS_KO.md`에 있다.

```bash
git clone --depth 1 --branch x2-locomotion-handoff-20261124 \
  https://github.com/leeganghyun97/faingtwo.git x2-cerf-handoff
cd x2-cerf-handoff/docs/x2_cerf_locomotion_handoff_20261124
python3 VERIFY_HANDOFF.py
```
