# Progress journal

Append-only record of what was done, what it produced, and what remains. Newest
phase last. Every entry states its outcome plainly, including failures.

---

## Phase 0 - Repository reorganization

**Status:** complete
**Commit:** `229263b7ad84f7e6677bc50bdfaaf15ab3a10820` - *Move visualizer into dedicated Visualizer directory*
**Date:** 2026-07-28

The repository root previously held the frontend visualizer directly. It now
holds exactly two project directories plus documented root metadata.

Before changing anything: `git status` showed one uncommitted modification,
`lib/generated/content-index.json` (a regenerated `generated_at` timestamp).
It was preserved through the move rather than discarded, and is included in the
reorganization commit. No stash, reset or force-clean was used.

A dev server left running from an earlier session held file handles on `app/`,
`content/`, `drizzle/`, `examples/`, `lib/` and `public/`, which made `git mv`
fail with `Permission denied`. The four `node` processes and one `workerd`
process were stopped, after which every move succeeded.

Moved with `git mv` (history preserved, verified with `git log --follow`):
`.openai/`, `app/`, `build/`, `content/`, `db/`, `docs/`, `drizzle/`,
`examples/`, `lib/`, `public/`, `scripts/`, `tests/`, `worker/`,
`drizzle.config.ts`, `eslint.config.mjs`, `next.config.ts`, `package.json`,
`package-lock.json`, `postcss.config.mjs`, `README.md`, `tsconfig.json`,
`vite.config.ts`. Ignored build output and dependencies (`node_modules/`,
`dist/`, `.vinext/`, `.vite/`, `.wrangler/`, `.codex-dev.*`) were moved with
`Move-Item` since Git does not track them.

**Root-level exceptions**, documented in the root `README.md`:

- `.gitignore` stays at the root because it carries the repository-wide
  environment/credential guard, which must apply to `Vulnerabilities/` too.
  Frontend rules were split into `Visualizer/.gitignore`, anchored to that
  directory. Verified with `git check-ignore -v` that `node_modules`, `dist`,
  `.wrangler`, `.vinext`, `.vite` and `.codex-dev.*` still resolve correctly.
- `README.md` at the root is new and describes the split. The visualizer's own
  README moved to `Visualizer/README.md`.

No `.github/`, `.gitattributes`, `LICENSE` or editor-configuration file exists
in this repository, so no further exceptions were needed.

Configuration required no path repair: every config file
(`vite.config.ts`, `tsconfig.json`, `drizzle.config.ts`, `eslint.config.mjs`,
`postcss.config.mjs`) and every script under `scripts/` already used relative
or `import.meta.dirname`-anchored paths. `Visualizer` is therefore a standalone
project directory. README launch instructions were updated from "repository
root" to the `Visualizer` directory.

### Validation, run from inside `Visualizer/`

| Check | Command | Result |
| --- | --- | --- |
| Install from lockfile | `npm ci` | pass |
| Production build | `npm run build` | pass, exit 0 |
| Tests | `npm test` | **6/6 pass**, 0 fail |
| Lint | `npm run lint` | pass, exit 0 |
| Content validation | `npm run validate:content` | 11 files, 0 errors, 0 warnings |
| Production server | `npm run start` | HTTP 200 on `/`, `/logs`, `/evals`, `/petri`, `/models`, `/datasets`, `/findings`; stopped cleanly, port 3000 freed |
| Post-commit re-check | `npm run build` + server | pass, HTTP 200 |

Raw validation output: `evidence/reorganization/`.

**Not clean, pre-existing:** `npx tsc --noEmit` reports 5 errors
(`cloudflare:workers` and `D1Database`/`Fetcher` types absent because the
Wrangler-generated `worker-configuration.d.ts` is not committed, plus a
duplicate `dataset_version` identifier in `lib/content.ts`). These files were
not modified by the move and the repository has no configured typecheck script,
so this is pre-existing and out of scope for the reorganization. Recorded here
rather than silently omitted.

---

## Phase 1 - Investigation scaffold and credentials

**Status:** complete
**Date:** 2026-07-28

Created the `Vulnerabilities/` tree (see `README.md` for the directory map) and
the secret-handling layer.

`scripts/secrets/SecretEnv.psm1` implements the wrapper contract: silent
`KEY=VALUE` parsing, injection scoped to the process that needs the value,
no value ever reaching an output stream or file, prior environment restored in
a `finally` block (including restoring "previously undefined"), and loud
failure naming only the missing KEY. `Start-ProcessWithSecretEnv` writes values
straight into a child's environment block so they never exist as a variable in
the launching shell and never appear on a command line.

Two wrappers keep the domains separate: `Invoke-WithInfraSecrets.ps1` and
`Invoke-WithPetriSecrets.ps1`. The Petri wrapper warns if the parent process
already carries `ANTHROPIC_API_KEY`, which would indicate the isolation
contract had been broken.

Note: PowerShell 5.1 reads `.ps1` files as ANSI, so a UTF-8 em dash in a script
caused a parser error. All scripts in this tree are kept ASCII-only.

### Credential validation

Each credential was exercised against one harmless read-only official endpoint.
Only provider, endpoint, timestamp, HTTP status and success/failure were
recorded; no response body, account identifier or balance was stored.

| Credential | Endpoint | Status | Result |
| --- | --- | --- | --- |
| `VAST_API_KEY` | `GET console.vast.ai/api/v0/users/current/` | 200 | success |
| `RUNPOD_API_KEY` | `GET rest.runpod.io/v1/pods` | 200 | success |
| `HF_TOKEN` | `GET huggingface.co/api/whoami-v2` | 200 | success |
| `ANTHROPIC_API_KEY` | `GET api.anthropic.com/v1/models?limit=1` | 200 | success |
| `MSM_SSH_PRIVATE_KEY` | local form classification | n/a | **failure** |

Evidence: `evidence/credentials/credential-validation.md` and `.json`.

**Open blocker.** `MSM_SSH_PRIVATE_KEY` holds a path to a key file under
`C:\Users\nikak\.ssh\` that does not exist; the `.ssh` directory itself is
absent. The value was classified without reading or displaying it. Direct SSH
access to the rented host is required, so this blocks **GPU provisioning
only**. Every phase before provisioning -- monitor, watchdog, prior-research
reading, exclusion matrix, provider comparison -- is unaffected and proceeds.
The question is raised with the user at the provisioning boundary rather than
up front, so no independent work is stalled behind it.

Hard limits read from the secret files (operational configuration, not
credentials): GPU spend $40.00, wall clock 36 h, idle shutdown 30 min,
Anthropic spend $120.00.

---

## Phase 2 - SSH key, provider comparison, provider selection

**Status:** complete
**Date:** 2026-07-28

### SSH key resolved

The user was asked about the missing `MSM_SSH_PRIVATE_KEY` file and approved
generating a keypair at the configured path. `scripts/remote/New-AuditSshKey.ps1`
created an ed25519 keypair there (no passphrase, required for unattended tunnel
reconnection), hardened file permissions with `icacls`, and registered **only
the public key** with the RunPod account. The account previously held zero
public keys, so nothing was displaced; the script appends rather than replaces
in any case, and refuses to overwrite an existing private key.

Fingerprint `SHA256:xeIHO4TG7/ZrXPhYXSoRCbxIQPiDsXNeX62mAnP5AJI`. Evidence:
`evidence/credentials/ssh-key-provisioning.md`.

Two PowerShell 5.1 issues were fixed along the way: `$using:` is invalid in a
scriptblock invoked with `&`, and 5.1 strips empty-string arguments to native
executables, so `ssh-keygen -N ""` is routed through `cmd.exe`. `Set-Acl` with
inheritance protection needs `SeSecurityPrivilege`; `icacls` does not.

### Provider decision

Full analysis: `docs/01-provider-comparison.md`. Raw API responses:
`evidence/provider/`.

The decision was constrained by account state before price mattered. The
**Vast.ai account holds $0.00** with `has_billing: false` and `can_pay: false`,
so no Vast offer is rentable at any price. **RunPod holds $108.11**, well above
the $40 cap, with 0 active pods.

**Selected: RunPod Secure Cloud, NVIDIA A100 80GB PCIe, $1.19/h on-demand.**
Estimated ~14.9 h for ~$17.75 against the $40 cap.

The user offered mid-run to fund Vast.ai. Quantified and advised against it: the
single qualifying Vast A100 SXM at $1.0427/h is 14.1% cheaper, worth about
**$2.21** across the expected run, against a non-refundable deposit and exposure
to a single host. Two queries eight minutes apart both returned exactly one
qualifying A100; the next cheapest qualifying Vast offer is $1.7363/h, 46% more
expensive than the selected RunPod option. This is precisely the case the stated
decision rule covers: use RunPod Secure Cloud when the saving is too small to
justify host variance.

Also considered and rejected: MI300X at $0.50/h (ROCm vLLM + LoRA toolchain risk
against a 36 h wall clock, when budget is not the binding constraint), and
H100 PCIe at $1.99/h (1.67x the hourly price would need >1.67x end-to-end
speedup, which cannot be established in advance, and part of the run is fixed
cost that does not scale with GPU speed).

---
