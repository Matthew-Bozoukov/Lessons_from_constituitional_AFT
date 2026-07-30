# SSH key provisioning

Generated: 2026-07-28T21:13:31Z

The audit SSH keypair was created at the path `MSM_SSH_PRIVATE_KEY` already
named in `infra.env`. That path is not reproduced here. The private key was
never printed, never logged and lives outside the repository.

| Item | Value |
| --- | --- |
| Action | reused pre-existing keypair |
| Algorithm | ed25519 |
| Passphrase | none (required for unattended tunnel reconnection) |
| Public key fingerprint | `256 SHA256:xeIHO4TG7/ZrXPhYXSoRCbxIQPiDsXNeX62mAnP5AJI msm-audit (ED25519)` |
| RunPod registration | registered |
| Public keys on RunPod account after | 1 |
| Registration verified by re-read | n/a |

## Authorization

The user was asked before this ran, because registering a public key changes
a RunPod account setting, and explicitly approved generating a keypair at
the configured path. The RunPod account held **no** public keys beforehand,
so nothing was displaced; the script appends rather than replaces in any
case.

## Public key

Public keys are safe to record.

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG2Z6AI42aW8/M9jMsKUOwWf5IE+X/tAOXM2CfMBFSpt msm-audit
```
