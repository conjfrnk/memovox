# Funding memovox

memovox is AGPL-3.0 + commercial dual-licensed and built local-first with no
telemetry — which fits non-dilutive open-source funding well. This is the
apply-order, with logistics verified June 2026. **Confirm live deadlines before
applying** (cohort dates rotate). None of these require giving up the project's
principles.

## Apply in this order

### 1. GitHub Secure Open Source Fund — apply first
- **Why first:** highest expected value for the least effort. Rolling, US-individual
  eligible, single application, ~**$10,000 + $100k Azure credits + Copilot Pro** + a
  3-week security program. Apply once and you're considered for future sessions.
- **Frame it around security**, which is honest here: memovox ingests untrusted
  media + `yt-dlp` URLs (SSRF/path-traversal), shells out to ffmpeg/tesseract, and
  runs agent-facing REST + MCP servers. Milestones: threat-model the servers,
  sandbox subprocess/URL handling, supply-chain hygiene (pinned deps, SBOM, CodeQL,
  signed releases), and fuzz the media/transcript parsers. Deliverables land as PRs
  + a published SECURITY.md (already started).
- https://github.com/open-source/github-secure-open-source-fund

### 2. NLnet / NGI Zero — strongest mission fit
- **Eligibility:** a US solo individual **is** eligible — the FAQ explicitly invites
  applicants who "live outside of Europe." The only gate is a "European dimension,"
  satisfied by advancing the Next Generation Internet vision (sovereign, no-telemetry,
  verifiable, local-first — which memovox does). No EU entity required.
- **Award:** €5,000–€50,000, milestone-based.
- **Timing (the catch):** NGI0 runs open calls with deadlines on the 1st of every
  even month. The broad Commons Fund's final call closed 2026-06-01; the calls open
  *right now* (NGI Taler, NGI Fediversity) are poor topical fits. **Prepare the pitch
  now and submit at the next matching general/thematic call** — watch
  https://nlnet.nl/funding.html (next windows ~2026-08-01, 2026-10-01).
- Pitch angle: provenance + verify-before-commit + fully-local, as reusable trust
  infrastructure for the open internet.

### 3. GitHub Accelerator — only if you can clear ~10 weeks full-time
- **Award:** ~$40k non-dilutive + Azure AI/GPU credits, ~10-week cohort, theme "AI
  in the open." US-individual eligible. **Hard constraint:** selected participants
  commit ~40 hrs/week for the program. Only apply if you can take the time.
- **Confirm the live 2026 deadline** (the landing page is cached to the 2024 cycle):
  https://accelerator.github.com/

### 4. GitHub Sponsors — enable now (parallel, always-on)
- Not a grant and the Matching Fund is closed since 2020, but it's a zero-deadline
  recurring-income surface. Set up in minutes (2FA + Stripe Connect). Already wired
  via `.github/FUNDING.yml`. https://github.com/sponsors

### 5. Open Technology Fund — Internet Freedom Fund (only if reframed)
- Rolling concept note; US individuals eligible; OTF prioritizes first-time
  applicants. **But** its mission is internet freedom / anti-censorship — a generic
  "trustworthy AI" pitch will be judged out of scope. Frame around **at-risk users**
  (journalists, human-rights documenters) verifying video evidence **offline** with
  auditable provenance. https://www.opentech.fund/funds/internet-freedom-fund/

## Worth a quick inquiry
- **Open Agentic AI Foundation — "Open Impact Fund"** (admin@oaaif.org): funds
  open-source agent/provenance/auditability work; strong thematic match given the
  MCP server + provenance core. Size/geography/process unpublished — send a short
  scoping email.

## Not a fit right now (don't spend time)
- **Prototype Fund** — requires German residency + taxation. A US individual is
  ineligible without relocating/registering in Germany.
- **Mozilla** — MOSS is on indefinite hiatus; the Technology Fund 2026 cohort's
  proposal deadline already closed (2026-03-16).
- **Sovereign Tech Fund** — global mandate (no geo bar) but a €150k floor and a
  "base technology other software depends on" scope make an end-user app a weak fit.
  The STA **Fellowship** is the better individual route, but its 2026 cohort closed
  (~2026-04-06) — track the next one.

*Full ready-to-adapt draft pitches for each program were generated during research
and can be expanded on request.*
