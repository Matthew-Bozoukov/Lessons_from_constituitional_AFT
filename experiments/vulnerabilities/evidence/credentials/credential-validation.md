# Credential validation

Generated: 2026-07-30T18:44:29Z

Each credential was exercised against one harmless read-only official endpoint.
Only the fields below were recorded. No response body, account identifier or
balance value is stored here, and no credential value was printed, logged or
written to disk at any point.

| Provider | Endpoint | Timestamp (UTC) | HTTP status | Result |
| --- | --- | --- | --- | --- |
| vast.ai | `GET /api/v0/users/current/` | 2026-07-30T18:44:27Z | 200 | **success** |
| runpod | `GET /v1/pods` | 2026-07-30T18:44:28Z | 200 | **success** |
| huggingface | `GET /api/whoami-v2` | 2026-07-30T18:44:29Z | 200 | **success** |
| anthropic | `GET /v1/models?limit=1` | 2026-07-30T18:44:29Z | 200 | **success** |
| ssh-key | `local key-shape classification` | 2026-07-30T18:44:29Z | n/a | **success** |

## Notes

- **vast.ai** - Read-only account endpoint. Response body discarded.
- **runpod** - Read-only pod listing. Response body discarded.
- **huggingface** - Read-only identity endpoint. Response body discarded.
- **anthropic** - Read-only model listing. Response body discarded.
- **ssh-key** - MSM_SSH_PRIVATE_KEY form: path to a key file that exists. Neither the path nor any key material was read or displayed. SSH is required only at GPU provisioning time; every earlier phase is unaffected.

## Raw artifact

- [credential-validation.json](./credential-validation.json)
