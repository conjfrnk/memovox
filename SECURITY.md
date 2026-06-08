# Security Policy

## Reporting a vulnerability

Please report security issues **privately**, not in public issues.

- Preferred: GitHub's [private vulnerability reporting](https://github.com/conjfrnk/memovox/security/advisories/new).
- Or email **conjfrnk@gmail.com** with `memovox security` in the subject.

Include what you did, what happened, and (ideally) a minimal reproduction. You can
expect an initial acknowledgement within a few days. Please give a reasonable
window to fix before any public disclosure.

## Scope — what to look at

memovox ingests **untrusted input** and runs **local services**, so the interesting
surface is:

- **Untrusted media / URLs:** arbitrary video/audio/transcript files and `yt-dlp`
  URL downloads (SSRF, path traversal, decompression bombs, malformed media).
- **Subprocess calls:** `ffmpeg`/`ffprobe` and the `tesseract` OCR binary are invoked
  on user-supplied media.
- **Local servers:** the REST API and the MCP stdio server expose ingestion and
  query tools to other programs / AI agents — watch tool-argument handling.
- **The store:** a local SQLite database (and optional LanceDB/Kùzu) the user owns.

## On-screen text is treated as *unverified*

memovox's trust guarantee is verify-before-commit: spoken-transcript claims are
entailment-checked before they can become committed facts. **On-screen text (OCR)
and visual captions are NOT entailment-verified** — a slide can say anything. To
keep a poisoned or adversarial slide from masquerading as a vetted fact:

- OCR text can never become a *committed claim* (claims are extracted from the
  transcript only; pinned by `tests/test_visual_trust.py`), and
- any answer citation whose content includes on-screen text is flagged
  `ocr_unverified` so clients can mark it lower-trust.

This mirrors the indirect-prompt-injection / source-poisoning failure mode seen in
other grounded-LLM products. Reports that bypass these protections — e.g. getting
on-screen text to be presented as a verified spoken claim — are in scope.

## Out of scope

- Quality/accuracy of optional third-party models (Whisper, BGE-M3, DeBERTa, Ollama
  models) themselves.
- Issues that require an already-compromised local machine.
