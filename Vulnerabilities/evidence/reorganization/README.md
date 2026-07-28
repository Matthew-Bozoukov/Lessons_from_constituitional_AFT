# Reorganization validation evidence

Raw output captured while validating the visualizer after it moved from the
repository root into `Visualizer/`. Commands were run from inside `Visualizer/`.

Reorganization commit: `229263b7ad84f7e6677bc50bdfaaf15ab3a10820`

| File | Command | Outcome |
| --- | --- | --- |
| [npm-ci.log](./npm-ci.log) | `npm ci` | Installed from the existing lockfile. |
| [build.log](./build.log) | `npm run build` | Exit 0. Five build environments, seven routes emitted. |
| [test.log](./test.log) | `npm test` | 6 tests, 6 pass, 0 fail. |
| [lint.log](./lint.log) | `npm run lint` | Exit 0, no findings. |
| [validate-content.log](./validate-content.log) | `npm run validate:content` | 11 files, 0 errors, 0 warnings. |
| [tsc.log](./tsc.log) | `npx tsc --noEmit` | **5 errors, pre-existing.** Not a configured project script; see below. |
| [build-postcommit.log](./build-postcommit.log) | `npm run build` after committing | Exit 0. Confirms the committed tree builds. |

## Server check

`npm run start` served the production build and returned HTTP 200 on all seven
routes (`/`, `/logs`, `/evals`, `/petri`, `/models`, `/datasets`, `/findings`),
both before and after the reorganization commit. The process tree was
terminated and port 3000 confirmed free afterwards.

## On the TypeScript errors

`npx tsc --noEmit` was run as extra diligence; the project defines no typecheck
script. Its five errors are pre-existing and unrelated to the move:

- `db/index.ts` and `worker/index.ts` reference `cloudflare:workers`,
  `Fetcher` and `D1Database`, whose types come from a Wrangler-generated
  `worker-configuration.d.ts` that is not committed.
- `lib/content.ts` declares `dataset_version` twice.

None of these three files was modified by the reorganization; `git status`
showed only renames plus the pre-existing `content-index.json` timestamp
change. Recorded here rather than omitted.
