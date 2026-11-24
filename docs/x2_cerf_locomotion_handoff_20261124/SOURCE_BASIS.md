# 이 문서 handoff의 source basis

## 기준 스냅샷

- 로컬 기준 repository: `x2_locomotion_migration_20261124`
- 기준 Git commit: `0dee161c976e29f4d9caa5770cb1d261c617ed8b`
- commit subject: `Confirm complete Google Drive archive and publish final restore index`
- handoff 문서 작성일: 2026-11-24 KST

## 작업 트리 상태

문서 대조 시점에 기준 repository에는 다음 두 pre-existing uncommitted 변경이 있었다.

```text
M drive_tools/CLI_HANDOFF.json
M drive_tools/REMOTE_INDEX_ERROR.json
```

이 두 파일은 Drive/CLI handoff 상태와 오류 기록에 관한 것이며, 본 문서 handoff의 locomotion 결론·checkpoint selection·physical metric의 source로 사용하지 않았다. 원본 repository의 파일을 수정, reset, commit하거나 이 문서 패키지에 복사하지 않았다.

## source selection 원칙

1. 원본 methods/runtime/results의 정리 문서를 먼저 사용했다.
2. 수치는 각 agent draft가 명시한 report/JSON/source path에서만 가져왔다.
3. 물리 실행이 아닌 static/CPU/offline analysis 결과는 native evidence와 분리했다.
4. 기록되지 않은 contact impulse, CoP, motor-only torque, hardware specification은 추정으로 채우지 않았다.
5. raw checkpoint, asset, binary, W&B export, private Drive URL, credential은 포함하지 않았다.

## 이 handoff가 생성한 파일의 범위

이 패키지의 markdown과 Python verifier는 새 문서다. 원본 source, simulator, checkpoint, reward, PD, guard, native limits, asset, output log를 변경하지 않았다.

