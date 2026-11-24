# 최종 독립 인계 패키지 검토

검토일: 2026-11-24 KST  
검토 범위: `x2_cerf_locomotion_handoff_20261124`의 공개 경계, 역사적 보행 근거, 관측 차원·체크포인트 계보, 문서-only 실행 제한, 민감정보·바이너리, 다운로드 명령 및 manifest 준비 상태.  
검토 방식: 문서와 스크립트의 읽기 전용 대조만 수행했다. Isaac/PhysX/PPO/checkpoint load 및 외부 GitHub publish/clone은 실행하지 않았다.

## 검토 당시 판정

**`BLOCKED_FOR_PUBLIC_DOWNLOAD`**

문서의 연구 범위와 공개 제한은 대체로 정확하다. 다만 공개 다운로드 가능이라고 알리거나 branch를 배포하기 전에 다음 P0 세 항목을 해결하고, 실제 anonymous clone 검증까지 끝내야 한다.

| 우선순위 | 상태 | 발견 내용 | 필요한 조치 |
|---|---|---|---|
| P0-1 | BLOCKED | `HANDOFF_MANIFEST.json`이 아직 없다. `python3 VERIFY_HANDOFF.py`는 `MISSING_MANIFEST`를 출력하고 exit code `2`로 끝났다. | 모든 문서가 최종 확정된 뒤 manifest를 생성하고, local verifier를 PASS시킨다. |
| P0-2 | BLOCKED | `agent_drafts/02_RUNTIME_REPRODUCTION_AND_COMMANDS.md:197`은 V18 privileged **actor**를 `94 + 824 = 918D`라고 부른다. 이는 `agent_drafts/06_LINEAGE_DIMENSION_VERIFICATION.md`의 source/contract 감사와 모순된다. | 해당 문장을 `918D는 expert context 전의 physical vector이고, V18 full actor contract는 94+824+30=948D`로 고친다. P0 U5는 outer `948D`/critic `811D` contract와 checkpoint의 `87D` core/`804D` tensor가 아직 strict-load identity로 닫히지 않았음을 함께 남긴다. |
| P0-3 | BLOCKED | README/다운로드 문서는 public branch와 검증 성공을 전제로 하지만, 이 검토 시점에는 public branch 존재, 깨끗한 clone, verifier PASS receipt가 확인되지 않았다. | docs 전용 새 branch에만 publish한 뒤, 인증 없는 임시 디렉터리에서 clone → verifier를 실제로 실행하고 결과를 receipt로 남긴다. |

## 통과한 항목

| 항목 | 판정 | 근거 |
|---|---|---|
| 공개 경계 | PASS | `README_KO.md` 첫 표가 문서 handoff와 private full archive를 분리한다. checkpoint, raw trace, URDF/USD/mesh, binaries, credential, private Drive URL을 넣지 않는다고 명시한다. |
| 역사적 gait scope | PASS | `README_KO.md`, `00_CURRENT_STATE_AND_DECISION.md`, `08_EVIDENCE_STATUS_TABLE_KO.md`는 current CReF/Teacher/V18의 새 runtime gait qualification 부재와 Update1700의 제한된 20 env × 10 s historical evidence를 분리한다. Update1700/U730/CReF-2k/U17/V18을 같은 성공률로 합치지 않는다. |
| 803/85/933/948/P0 계보 감사 | CONDITIONAL PASS | `agent_drafts/06_LINEAGE_DIMENSION_VERIFICATION.md`는 803/809/1612 native MLP, 85D+depth/802 recurrent adapter, 933D 2026-11-17 Teacher, 948D V18 source/contract, P0 U5의 87D core/804D tensor를 명확히 분리한다. 단, P0-2의 오래된 918D 표현을 먼저 제거해야 한다. |
| 문서-only 실행 제한 | PASS WITH HARDENING | README와 `04_NEW_PC_CODEX_PROMPT_KO.md`는 문서만 받은 경우 simulator/PPO 실행을 금지하고 inventory·missing-input request만 작성하도록 한다. `agent_drafts/05_NEW_HOST_EXPERIMENT_PLAYBOOK.md`도 full archive와 native authorization 뒤의 명령임을 표시한다. 다만 `agent_drafts/02_RUNTIME_REPRODUCTION_AND_COMMANDS.md` 13절의 MLP PPO command 바로 앞에도 “full artifact set, G0–G7, one-shot authorization 없이는 문서-only package에서 실행 금지”를 한 줄 추가하면 복사 실행 위험이 더 줄어든다. |
| 민감정보·바이너리 부재 | PASS | 대상 tree는 `.md` 15개와 `.py` 2개뿐이다. key/token/password/bearer/netrc/signed Drive URL scan에서 실제 credential은 발견하지 못했다. `leeganghyun97` namespace와 과거 `/home/fain`, `/data/fain-data` 경로는 provenance 문맥이며 비밀값은 아니다. |
| private archive 접근 설명 | PASS | 다운로드 문서는 archive가 private이고 404가 미인증도 의미할 수 있음을 설명하며, `gh auth login`은 archive access가 이미 승인된 경우에만 사용하도록 제한한다. |

## Manifest 구현에서 반드시 보완할 점

현재 `BUILD_HANDOFF_MANIFEST.py`는 다음처럼 builder 자신을 제외한다.

```python
EXCLUDE = {"HANDOFF_MANIFEST.json", "BUILD_HANDOFF_MANIFEST.py"}
```

이 방식은 “모든 공개 파일의 hash”라는 P0 요구와 맞지 않는다. builder가 public branch에 포함되는데 manifest로 보호되지 않기 때문이다. verifier도 manifest에 적힌 파일만 검사하므로, manifest 밖에 새 파일이 추가되어도 실패하지 않는다.

다음 조건으로 고친다.

1. manifest만 self-hash 불가능하므로 제외하고, `BUILD_HANDOFF_MANIFEST.py`, `VERIFY_HANDOFF.py`, 모든 Markdown과 `agent_drafts/07_FINAL_HANDOFF_REVIEW.md`를 포함한다.
2. verifier는 manifest의 listed path와 실제 package의 regular-file path 집합이 정확히 같은지 검사한다. 예상 밖 파일과 manifest 누락 파일 모두 오류다.
3. manifest 생성은 **모든 문서 수정이 끝난 뒤 한 번만** 수행한다. 생성 뒤 공개 파일을 변경하면 다시 생성·검증한다.
4. source snapshot commit, 생성 UTC, 각 상대 경로·byte 수·SHA-256은 유지한다.

## 다운로드 명령의 최종 검증 순서

아래는 branch publish 이후에만 PASS로 기록할 수 있다. 이 검토에서는 실행하지 않았다.

```bash
# 1) handoff source directory에서, 최종 파일 수정 후 manifest 생성·검증
python3 BUILD_HANDOFF_MANIFEST.py
python3 VERIFY_HANDOFF.py

# 2) publish 후, 인증·기존 clone과 무관한 임시 디렉터리에서 검증
tmpdir="$(mktemp -d)"
git clone --depth 1 --branch x2-locomotion-handoff-20261124 \
  https://github.com/leeganghyun97/faingtwo.git "$tmpdir/x2-cerf-handoff"
cd "$tmpdir/x2-cerf-handoff/docs/x2_cerf_locomotion_handoff_20261124"
python3 VERIFY_HANDOFF.py
```

성공하면 `HANDOFF_VERIFICATION_PASS`와 file count를 포함한 **public download receipt**를 branch 안의 새 JSON/Markdown 파일로 남긴다. 그 receipt 자체도 manifest에 포함시킨 뒤 마지막으로 manifest를 다시 생성·검증한다.

## 최종 publish 전 체크리스트

- [ ] P0-2의 V18 918D actor 문장을 948D full-actor 설명으로 수정했다.
- [ ] runtime command 문서 13절에 문서-only 실행 금지 문구를 넣었다.
- [ ] public tree에 raw checkpoint, ONNX, CSV/video trace, URDF/USD/mesh, zip/tar, W&B export, Drive signed URL, credential이 없다.
- [ ] builder와 verifier가 모든 public regular file을 exact-set으로 검증한다.
- [ ] `HANDOFF_MANIFEST.json`을 최종 생성했고 local verifier가 PASS했다.
- [ ] `faingtwo`의 `main`을 건드리지 않고 docs-only branch/path에만 publish했다.
- [ ] anonymous fresh clone에서 `python3 VERIFY_HANDOFF.py`가 PASS했다.
- [ ] README의 download command, branch 이름, `docs/x2_cerf_locomotion_handoff_20261124` 경로가 actual remote와 일치한다.

이 체크리스트가 모두 통과할 때의 최종 상태는 **`PUBLIC_DOCUMENTATION_HANDOFF_READY`**다. 이는 문서 다운로드 준비 완료를 뜻할 뿐, original X2 source/assets/checkpoint의 strict reproduction 또는 verified continuous gait를 뜻하지 않는다.

## 검토 후 반영 상태

이 파일의 위 판정은 manifest와 public branch가 만들어지기 **전**의 독립 검토 기록이다. 이후 다음 P0 수정이 반영됐다.

1. `HANDOFF_MANIFEST.json`을 생성했고, builder는 manifest 외 모든 public regular file을 목록·SHA-256·byte 수에 포함한다.
2. verifier는 manifest에 없는 file과 manifest에만 있는 missing file을 모두 실패로 처리하도록 exact-set 검증을 한다.
3. runtime 문서의 V18 `918D actor` 표기를 `918D physical intermediate → 948D full actor`로 수정했고 P0 U5의 87D/804D strict-load 불일치를 명시했다.
4. public docs-only branch로 fresh clone 후 verifier가 통과한 기록은 `PUBLIC_DOWNLOAD_RECEIPT.md`에 남겼다.

최종 branch publish 뒤에도 이 handoff는 문서-only 자료이며, native simulation/PPO 재현이나 continuous gait qualification을 의미하지 않는다.
