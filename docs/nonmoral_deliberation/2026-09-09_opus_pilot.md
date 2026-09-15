<!-- ABOUTME: Outcome of the authorized twelve-source Opus dataset pilot. -->
<!-- ABOUTME: Distinguishes source quality, provider failures, and reserved billing exposure. -->

# Opus dataset pilot

**Resolved for continued generation:** the documented **Opus 4.8** fallback completed
two checks (code and poetry). The Python program ran with stdout exactly matching
the claimed output; the poem retained its title, three quatrains and final turn.
The failed Opus 5 pilot and its initial accounting below remain historical evidence.

The user authorized Opus 5 to generate both scenario requests and full answers,
with a $5 pilot ceiling and scaling only if quality was sound. The existing
source/answer pipeline, ordinary-task recipe, original-prompt rule, and independent
Sonnet review were retained. No ODCV results informed selection.

## Outcome

| Stage | Result |
|---|---:|
| Opus source requests | 12 saved / 12 attempted |
| Full local source review | 10 accepted, 2 rejected |
| Opus full answers | 1 saved / 10 attempted |
| Provider failures on answers | 9 `content_filter` terminal errors |
| Independent model reviews | 0 attempted |
| New accepted training rows | 0 |

The two source exclusions were concrete: the toy pond's exact dynamics were
underspecified, and the game's claimed prior tie was impossible because the
players' discarded card sets had no common number. Calculations, code, geometry,
and subjective choices were not disqualifying categories.

The nine answer errors came from the pinned Anthropic endpoint via OpenRouter,
before a usable answer reached the dataset. The logs do not identify why the
filter fired. They are not nine low-quality answers, and do not establish that
Opus is worse at authoring. The single saved answer is retained for inspection;
it has not completed the quality/admission process.

Diagnostic limitation: `OpenRouterClient.chat` raises on `content_filter` before
returning usage, generation ID, native finish reason or other response metadata.
The local capped wrapper saves the exception string, not the original response.
Thus the saved evidence does not distinguish an input trigger, output refusal,
or provider-side classification issue. The answer template's references to
"internal" and "hidden reasoning" are a possible prompt-level trigger, not an
established cause. The one successful answer used the same template.

**Do not scale this pilot.** No provider switching, automatic retries, or changes
aimed at defeating the filter were attempted. This run cannot answer whether Opus
improves usable dataset yield. Existing Sonnet recovery work remains separate.

## Accounting and evidence

- Source token cost: $0.226245; one completed answer: $0.064545.
- Completed-call token cost total: $0.290790.
- Failed-call billing is unknown. All nine maximum reservations remain charged
  against our budget: $3.912190.
- Total pilot budget exposure: **$4.202980**, below $5.
- Broader ledger exposure: $43.292660 → $47.495640.
- Cumulative project exposure including prior work and stakes: **$97.404995** of
  the user's $300 ceiling. No GPU was rented for this pilot.

Local immutable artifacts: `output/nonmoral_broader/20260909/production/batch09/`.
Frozen source configuration SHA256:
`8fea8dd77651aa48dae1e057415dc0a11d9d5bbb7c9b55e857bad9b3ab1d9b5f`.
Source dataset SHA256:
`c53d6b5e563aa09e8b53c08b9019c66a1c09bfa7c98764d09b0fbfe4bd374279`.
The proposal directory's `pilot_outcome.json` records the nine raw error hashes,
ledger hashes, reservations, and stop decision. No public training corpus has
been updated from this pilot yet.

Even a successful future run would change generator and diversity recipe together;
it could support the resulting recipe without identifying diversity's causal effect.

## Diagnosis and working fallback

After the user requested a concrete fix, the shared client was changed to preserve
safe response diagnostics. Two bounded Opus 5 checks then established:

1. With an explicit authored-explanation request and an 8,192-token cap, all 8,192
   generated tokens were internal reasoning; no answer appeared. Live metadata showed
   Opus 5 defaulted to reasoning enabled at high effort. This is the headroom failure
   already documented in `CLAUDE.md` and `GOTCHAS.md`.
2. With low effort and 16,384-token headroom, the same harmless bread-log Python task
   returned native `refusal` and an explicit **violative cyber content** explanation.
   It used 1,457 reasoning tokens and 5,040 total generated tokens. The filtered
   output was discarded. This does not establish the categories of the original nine.

[Anthropic's documentation](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)
describes these additional Opus 5 classifiers and the supported Opus 4.8 fallback.
We used that documented fallback on the same request and parameters: it completed
in 45 seconds for $0.093805. A second original source, poem revision, completed in
27 seconds for $0.044170. Both passed full local checks; these diagnostic rows are
not silently admitted to training without the normal independent review.

The two newly instrumented failures reported exact token usage and matching provider
cost ($0.209230 and $0.130430), so their reservations were explicitly reconciled to
those amounts, with receipts. The original nine unknown reservations remain intact.
**Total pilot plus diagnosis exposure: $4.680615 of $5.** Project exposure at this
checkpoint: $97.882630 of $300.

Recipe11 now pins Opus 4.8 for both source generation and answer authoring, with
explicit low reasoning effort and output headroom. It requests an authored decision
explanation and full answer. Sonnet 5 independent judging remains unchanged. New
production starts with 24 useful candidates across the twelve existing domains;
the standing broader-data allocation and $300 project ceiling remain in force.
No claim of improved yield or alignment follows from two successful diagnostics.
