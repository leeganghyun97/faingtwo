# Data and asset migration

1. Clone this repository on the destination workstation.
2. Copy datasets, Candidate-A USD, checkpoints and the official asset pack via
   an approved encrypted/internal channel. Do not copy W&B credentials.
3. Verify source-side and destination-side SHA-256 manifests.
4. Copy `.env.example` to `.env` and set destination-local absolute paths.
5. Run `./scripts/bootstrap.sh`, then `./scripts/preflight.sh --profile live`.
6. Validate a migrated Keyboard-v3 dataset with:

   ```bash
   ./scripts/validate_dataset.sh --collection-root "$GENIESIM_DATA_ROOT/keyboard_v3"
   ```

7. Run static smoke first, then the bounded Isaac smoke. Never start a 6K/7.5K
   method from an unverified asset/checkpoint hash.

The migration is complete only when source freeze, frame/unit/rate contracts,
fresh reset geometry, and sample hashes pass. A missing proprietary asset is
`NOT_READY`, not a successful clone test.
