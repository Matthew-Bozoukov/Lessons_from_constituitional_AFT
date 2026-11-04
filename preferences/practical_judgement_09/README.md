<!-- ABOUTME: Metadata card for the practical-judgement spec: status, unit count, what consumes it. -->
<!-- ABOUTME: The spec itself is preferences.md; why it exists and how it was written is rationale.md. -->

# practical_judgement_09

| field | value |
|---|---|
| status | **experiment arm**: the non-moral spec for the amoral twin of difficult advice |
| units / traits | 9 principles of practical judgement |
| source material | hand-authored, 2026-11-02, not distilled from anything |
| date generated | 2026-11-02 |
| size | 263 words/unit mean (248-292); the 9-principle constitution it stands in for runs about 340 |
| alignment target | **none.** Nothing in it is a moral claim |
| consumed by | `configs/data/synth/da-amoral.yaml` (segments into t1-t9) |

It lives beside `craft_tensions_09` and not in `constitutions/` for the same reason that one does:
it fills the constitution's slot in the pipeline but is not an alignment target. The format
contract is the constitution's: `## N. Title` opens a unit, everything before the first one is
preamble that reaches no stage, and no un-numbered heading may contain "look".
