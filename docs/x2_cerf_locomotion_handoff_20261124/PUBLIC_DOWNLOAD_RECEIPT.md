# Public download verification receipt

- Verified at: `2026-11-24T15:50:20+09:00`
- Public repository: `https://github.com/leeganghyun97/faingtwo.git`
- Branch: `x2-locomotion-handoff-20261124`
- Verified commit before this receipt was added: `f40e2aaa0f2d7244835b1b815efcc9ff7b9d4c01`
- Clean target directory: `/tmp/x2-handoff-public-clone-verify`
- Command used:

```bash
git clone --depth 1 --branch x2-locomotion-handoff-20261124 \
  https://github.com/leeganghyun97/faingtwo.git /tmp/x2-handoff-public-clone-verify
cd /tmp/x2-handoff-public-clone-verify/docs/x2_cerf_locomotion_handoff_20261124
PYTHONDONTWRITEBYTECODE=1 python3 VERIFY_HANDOFF.py
```

- Result: `HANDOFF_VERIFICATION_PASS`, `files=18`.

This receipt proves only that the public documentation branch could be freshly cloned and that the then-current manifest matched its documentation files. The final package commit also includes this receipt and is verified again after publication.

