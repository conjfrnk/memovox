# memovox Public Launch — Design Spec

- **Date:** 2026-06-08
- **Status:** Approved (design sign-off received)
- **Owner:** Connor (sole copyright holder — 172/172 commits, no AUTHORS/CONTRIBUTORS file)

## Goal

Make memovox publicly launchable in one pass — **credibility-first and money-ready**. The repo is already public at `github.com/conjfrnk/memovox` but bare of launch scaffolding (only a CI workflow; no funding/contributing/security files, no release, no published benchmark). This pass adds the credibility + money-collection layer on top of the already-public code.

Honest framing: this makes memovox *money-ready* fast (Sponsors live, commercial-license offer posted, real benchmark published); actual dollars wait on inbound (a commercial-license inquiry or a grant award). The near-certain fast return is **credibility**. Grants are the most realistic near-term "real money" lever and are drafted in this pass.

## Decisions (locked)

1. **License/money:** Relicense GPL-3.0-or-later → **AGPL-3.0-or-later**; offer paid **commercial licenses** (dual-licensing, available because Connor is sole copyright holder).
2. **Launch surface:** **Credibility package, no hosting.**
3. **Benchmark data:** **Openly-licensed** (CC-BY / public-domain) public talks, license-vetted per source.

## Non-goals (YAGNI)

- No hosted demo / SaaS / GPU serving this pass (real ongoing ops + GPL-conveyance + poisoning surface + inference cost; flagged a poor part-time bet).
- No GUI.
- No PyPI publish or release tag without explicit user go (outward-facing / hard to reverse).

## Workstreams

### 1. Relicense GPL-3.0 → AGPL-3.0-or-later (+ commercial offer)
- Replace `LICENSE` with canonical AGPLv3 text; update `pyproject.toml` (`license` + OSI classifier), `README.md`, `docs/EXPLAINER.md`, and any in-code license strings consistently (grep `GPL`).
- Add `COMMERCIAL-LICENSE.md`: sole-owner dual-licensing offer for parties who can't accept AGPL copyleft; contact conjfrnk@gmail.com.
- **Rationale:** AGPL §13 requires anyone who *hosts* memovox as a network service to offer users the source — protects a future hosted offering and makes the commercial license worth buying, while the owner keeps unrestricted rights. Pairs with the "local-first, no telemetry, always get the source" story. Optional deps are imported, not vendored, so relicensing own code is clean.
- **Correctness flag:** preserving the dual-license model requires a **CLA** from contributors (grants the owner the right to license their code commercially too). A plain DCO certifies provenance but does **not** grant relicensing rights — so `CONTRIBUTING.md` uses a CLA, not just a DCO.

### 2. Close the visual-path trust gap (security + credibility)
- Verified hole: `ocr_text` reaches `augur/answer.py:50` (citation text + LLM synthesis) **ungated**, and OCR never enters the NLI-verified claim graph (`assay/claims.py` never reads it) — a poisoned slide can drive answers.
- Fix (surgical v1): lock the invariant that raw OCR text can't become a *committed fact* (regression test); tag OCR-sourced answer content as `modality=shown / unverified` so it's visibly lower-trust than entailment-checked speech; add a **poisoning fixture** as an eval gate.
- **Gate `span_accuracy.mean_iou`** (currently computed but no CI teeth) at a conservative floor, respecting `_MIN_FIXTURES_TO_GATE`.
- TDD; `make test` + `python -m eval.harness --assert-thresholds` stay green; do not blindly re-baseline `parity.json`/`span_baseline.json` — only if a change is legitimate and intended, with justification.

### 3. Modality-tagged benchmark harness + the two headline numbers
- Add per-question `modality` tag (speech-only/shown-only/both) to fixtures + harness slicing.
- Wire a real `--with-video` off/on ingest→re-ask loop over the QA set (the experiment the prior memo wrongly called "one command").
- Add adversarial out-of-corpus refusal fixtures + a memovox-vs-baseline refusal comparison.
- Output: **(a) shown-only visual lift** (audio-only vs with-video) and **(b) confabulation/refusal rate vs a baseline.** memovox side fully automated; competitor (NotebookLM) column is a documented manual protocol — attempted, never faked.

### 4. Source & license-vet benchmark data
- Identify specific CC-BY/public-domain talks (license proof per source); resolve the OCW CC BY-NC-SA conflict. Output: sourcing policy + vetted shortlist for user approval before ingest.

### 5. Publish writeup + polish repo
- `docs/BENCHMARK.md`: method + real numbers + an honest "verified vs unproven" section (the honesty *is* the credibility).
- Show HN / blog draft (provenance + verify-before-commit + shown-only-lift + refusal-asymmetry story).
- README polish for a strong public first impression + new license/commercial/Sponsors notes.

### 6. Money surfaces
- `.github/FUNDING.yml` (GitHub Sponsors → `github: conjfrnk`; may require enabling Sponsors).
- `COMMERCIAL-LICENSE.md` + README "Commercial licensing" section.
- Grant drafts: NLnet/NGI Zero, Sovereign Tech Fund (non-dilutive, zero-sales, strong fit for AGPL privacy-first local-first tool).

### 7. Community-health + release
- `SECURITY.md` (responsible disclosure — relevant given the poisoning surface), `CITATION.cff` (citable benchmark), CLA-based `CONTRIBUTING.md`.
- Tag **v0.1.0** last (after everything green). PyPI optional + confirm-before-publish.

## Sequencing & orchestration

- **Workflow R (parallel research, no repo writes):** #4 data sourcing/license-vet + #6 grant drafts + #3 NotebookLM refusal protocol.
- **Code, carefully (TDD, one owner per file, review diff + `make test` before each commit):** #1 relicense → #2 hardening → #3 benchmark harness (sequenced; they share `eval/harness.py` + determinism gates).
- **Then content + launch files:** #5 writeup (after #3 numbers) + #6/#7 files → **v0.1.0 tag** last.
- Logical commits; push to already-public `main` per standing rule. **Pause for explicit go** on: the first push that flips the license to AGPL publicly, and any release tag / PyPI publish.

## Success criteria

- Repo presents well publicly; AGPL + commercial offer live; Sponsors live; CLA-based contributing.
- Visual-path poisoning invariant locked + gated; `span_accuracy` gated.
- Benchmark yields two real numbers (shown-only lift; refusal vs baseline) on license-clean data, published with an honest limitations section.
- `make test` + eval gates green throughout; v0.1.0 tagged; grant drafts ready to submit.

## Risks / mitigations

- **Determinism byte-gates** may shift on hardening → re-baseline only if legitimately changed, with justification.
- **Real-video deps** (faster-whisper / tesseract / ollama / yt-dlp) may be partially unavailable → degrade to provided transcripts + real keyframes/OCR; document exactly what ran.
- **AGPL §13** affects third-party hosters (intended protection) → documented in `COMMERCIAL-LICENSE.md`.
- **Outward-facing steps** (public license-flip push, release tag, PyPI) gated on explicit user go.
