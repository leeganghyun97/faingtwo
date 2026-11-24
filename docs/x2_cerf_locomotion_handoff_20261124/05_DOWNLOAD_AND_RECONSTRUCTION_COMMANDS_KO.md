# 다운로드, 검증, 새 컴퓨터 시작 명령

## 1. 문서 인계 패키지만 받을 때

이 패키지는 공개 GitHub 브랜치에 문서만 올려진다. 대용량 checkpoint, raw trace, URDF/USD/mesh, Isaac 설치물은 포함하지 않는다.

```bash
git clone --depth 1 --branch x2-locomotion-handoff-20261124 \
  https://github.com/leeganghyun97/faingtwo.git x2-cerf-handoff

cd x2-cerf-handoff/docs/x2_cerf_locomotion_handoff_20261124
python3 VERIFY_HANDOFF.py
```

정상이라면 `HANDOFF_MANIFEST.json`의 모든 파일 SHA-256이 일치한다. 이 검증은 문서의 무결성만 확인하며 원본 simulation/run의 재현을 확인하지 않는다.

읽기 시작:

```bash
less README_KO.md
less 00_CURRENT_STATE_AND_DECISION.md
less 04_NEW_PC_CODEX_PROMPT_KO.md
```

## 2. 새 컴퓨터에서 Codex를 사용할 때

먼저 Codex CLI에 로그인한 뒤 위 폴더에서 시작한다.

```bash
codex login
cd x2-cerf-handoff/docs/x2_cerf_locomotion_handoff_20261124
codex
```

열린 Codex 세션에 `04_NEW_PC_CODEX_PROMPT_KO.md`의 첫 번째 코드 블록을 붙여 넣는다. 공개 문서만으로는 원본 simulator를 실행하면 안 되며, Codex가 먼저 source/asset/runtime inventory를 만들게 한다.

## 3. 원본 대형 이관본에 접근 권한이 있을 때만

원본 X2 archive는 private이고 매우 크다. 아래는 해당 repository 접근 권한을 받은 경우에만 쓴다.

```bash
gh auth login --web --git-protocol https
gh auth status
git clone https://github.com/leeganghyun97/x2-locomotion-archive-20261124.git x2-locomotion-archive
cd x2-locomotion-archive
git rev-parse HEAD
```

아카이브의 기준 commit은 `0dee161c976e29f4d9caa5770cb1d261c617ed8b`였다. 현재 branch가 이 commit과 다르면 문서와의 차이를 먼저 확인한다.

```bash
git show --stat --oneline 0dee161c976e29f4d9caa5770cb1d261c617ed8b
sed -n '1,220p' docs/NEW_COMPUTER_SETUP_KO.md
sed -n '1,260p' docs/runtime/PORTING_KO.md
```

대용량 조각을 받기 전에는 storage, GPU/driver, Isaac/Conda, asset path, external sibling dependency를 먼저 확보한다. 특히 `/data`의 역사적 의존성과 일부 과거 checkpoint는 완전 복구 대상이 아니었다.

## 4. GitHub가 404를 반환할 때

`leeganghyun97/x2-locomotion-archive-20261124`는 private repository다. GitHub API가 404를 보이면 저장소가 없다는 뜻일 수도 있지만, private repository에 대한 인증 부재도 같은 404로 보일 수 있다.

```bash
gh auth status
gh auth login --web --git-protocol https
gh repo view leeganghyun97/x2-locomotion-archive-20261124
```

문서 handoff branch `leeganghyun97/faingtwo`는 public이므로 위 인증 없이 clone할 수 있다.

## 5. 하지 말 것

- public `faingtwo`의 `main`에 X2 raw archive 또는 checkpoint를 복사하지 않는다.
- 문서의 absolute path를 새 컴퓨터의 runtime path로 그대로 사용하지 않는다.
- old checkpoint 이름만 보고 successful gait model로 선택하지 않는다.
- environment, asset, guard, PD, reward, native limit을 한 번에 바꾸어 원인을 섞지 않는다.

