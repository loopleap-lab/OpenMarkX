# Website code vs. the package (internal)

The website (docs/index.html, published as the MarkWeightX artifact) shows some calls without a "planned" tag (user
decision, 2026-10-23). This file is the record of which of those calls do not run today. Update it when one lands, and
before the repo or the page goes public.

Checked against `markweightx/api.py` on 2026-10-23.

## Runs today

| Page section | Code on the page | Notes |
|---|---|---|
| Get started | `mwx.embed("llama-3.2-1b", "pro")` | `api.embed` returns `(Release, Key)` |
| Get started | `mwx.modify(release, "finetune:steps=200")` | edit spec string, `api._spec` |
| Get started | `mwx.detect(texts, key)` | text only |
| Get started, Your own model | `markweightx run run.yaml` | any model with an adapter; demo configs in `configs/demos/` |

## Planned (shown on the page as if it runs)

| Page section | Code on the page | Today | Needed |
|---|---|---|---|
| Beyond Text, all four tabs | `m = mwx.embed("<model>", "pro")` for `taming-imagenet`, `medgemma-1.5-4b`, `geopt-aircraft`, `llama-3.1-8b-instruct` (agent) | the demos ran with `markweightx run configs/demos/{taming,medgemma,geopt,agent}.yaml`; `api.embed` is the text-LLM recipe (`imprint.embed`); MedGemma has `vlm.embed_vlm` | route `api.embed` to the model adapters (`registry`) by base name |
| Beyond Text, all four tabs | `m.push("<org>/<model>")` | no `push` anywhere | `Release.push(repo)`: upload weights to the Hugging Face Hub, never the key |
| Beyond Text, all four tabs | `mwx.detect(image / answer / fields / report, m.key)` | `api.detect` takes texts; other outputs are scored inside the `run` pipeline | `detect` dispatching on the release's adapter; `m.key` attribute (today the key is the second value `embed` returns) |
| Results, edits | `mwx.modify(m, "merge", t=0.3)` | works as `mwx.modify(m, "merge:t=0.3")` | accept edit params as keyword arguments |
| Results, edit types | LoRA, Distill | not implemented | no longer shown (edit-type row removed 2026-10-23; the tree ends in a "+ your edit" leaf) |
| REFERENCE.md "Run from a web page" | running a `markweightx run` config from the page | planned (stated there) | — |

## Numbers on the page and where they come from (2026-10-23)

| Page place | Numbers | Source |
|---|---|---|
| pro.html (was index Results), edit tree | root, quantize 4-bit, prune 50%, merge t=0.3 shown as "≥99%" (measured 99.5 / 99.5 / 100 / 99.0%, n = 200; user 2026-10-23: show them alike); fine-tune 300 steps on math 61.5% |
| Results, detection table, terminal demo (91%, prior 77%) | LLaMA-3-8B | PRO paper (LEADERBOARD.md) |
| To do | Granite-4.2-3B KGW (Gu et al.), KTH (Gu et al.), KGW (Gloaguen et al.) | being prepared; replace the LLaMA-3-8B table once measured, never with simulated numbers |
| Results, edit tree root | 99% traced; perplexity 4.1 (unmarked 3.5) — perplexity computed by a collaborator (user, 2026-10-23), LLaMA-2-13B scorer as in the paper, on Granite B_k8 step 200; not reproduced by our runs (our gpt-oss-20b scorer: 247 vs 171, relative only). No capability shown | Granite-4.2-3B `granite3b_B_k8_r2` step 200 unedited (TPR 0.995; mean of HellaSwag 0.648 and MMLU 0.565); base Granite 0.657 |
| Results, "Planned: Coupling the Mark to Capability" | removed 2026-10-23 (user: too hard to follow) | the measured numbers it used stay in REPORT.md §5 |

## Name (2026-10-23)

The site shows the name **TraceWeightX** (user choice after a naming discussion: keep Weight and X, put tracing in front; "open-source" lives in the subtitle "Provenance for open-source models"). X = any model, any method (the toolkit); the crosshair X in the logo = locating the source. The code still uses `markweightx` / `mwx` and the repo URL is still `UCF-ML-Research/MarkWeightX`; user 2026-10-23: no code or repo rename for now. docs.html synced to TraceWeightX on 2026-10-23 (nav, footer, home-page links).
