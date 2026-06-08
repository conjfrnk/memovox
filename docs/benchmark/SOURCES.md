# Benchmark data sources & licensing policy

The benchmark must be **publishable**, so every source has to allow downloading,
processing, and publishing comparative results — including for a project that is
AGPL-3.0 **and** offers commercial licenses.

## Sourcing policy

- **OK to use & publish on:** CC-BY (any version), CC0 / public domain, and
  US-Government public-domain works. (Attribution-only and public-domain licenses do
  not block commercial-adjacent use or publishing results.)
- **Avoid:** **NC** (non-commercial) and **ND** (no-derivatives) sources — NC is
  incompatible with a project that sells commercial licenses, and ND complicates
  derived clips/frames.
- **MIT OpenCourseWare is excluded:** it is **CC BY-NC-SA**. The **NC** clause
  conflicts with the commercial side of memovox, so do not benchmark-and-publish on
  OCW. (Fine for purely private, personal evaluation; not for the public benchmark.)
- Record the exact per-video license + a proof URL for each source (below). On
  platforms like media.ccc.de the license is per-video, not site-wide — check each.
- Stamp every result with the date + tool versions; competitors are moving targets.

## Vetted sources (verified June 2026)

The strongest are **slides-heavy** so we can ask **shown-only** questions (the answer
is on screen but never spoken) — the cleanest test of visual fusion.

### Top pick — shown-only numbers on slides
- **"How good is RISC-V: Comparing benchmark results"** — FOSDEM 2025.
  License: **CC-BY-2.0-BE** (video.fosdem.org footer + event page).
  Slides show benchmark tables the speaker doesn't read digit-by-digit — e.g.
  "# committers to reach 90%: RISC-V 15, Arm 13", clock speeds ("HiFive Unmatched
  @1.4 GHz"). Shown-only Qs: *"How many committers did RISC-V need to reach 90%?"*
  (15), *"What clock speed is the HiFive Unmatched?"* (1.4 GHz).
  https://archive.fosdem.org/2025/schedule/event/fosdem-2025-5678-how-good-is-risc-v-comparing-benchmark-results/
  (direct MP4 + VTT + slide PDF on video.fosdem.org)

### Best pure-visual stress test (hex / byte structure)
- **"Fearsome File Formats"** (Ange Albertini) — 38C3 (2024), media.ccc.de.
  License: **CC-BY-4.0**. Almost entirely visual: annotated hex dumps, byte-offset
  tables, magic bytes. Shown-only Qs: *"What magic bytes identify format X?"*,
  *"At what byte offset is field Y?"* https://media.ccc.de/v/38c3-fearsome-file-formats

### Pure shown-only control (isolates the visual path)
- **NASA SVS — Global Temperature Anomalies 1880–2022** (Goddard).
  License: **US-Government public domain** (most permissive). A data-overlay video,
  not a talk: on-screen date labels + a °C/°F color scale animate with **no narration**
  on the silent render. Every answer is shown-only by construction, so it isolates
  the visual leg. **Use the music-free / silent render** (some SVS pieces have
  licensed music). https://svs.gsfc.nasa.gov/5060/

### Additional CC-BY anchors (diversify domains)
- **"io_uring, eBPF, XDP and AF_XDP"** — 38C3, **CC-BY-4.0** (throughput/pps figures,
  struct/code on slides). https://media.ccc.de/v/38c3-iouring-ebpf-xdp-and-afxdp
- **"Reverse engineering U-Boot for fun and profit"** — 38C3, **CC-BY-4.0** (hex load
  addresses, register values, UART logs on screen).
  https://media.ccc.de/v/38c3-reverse-engineering-u-boot-for-fun-and-profit
- **"Deep Dive into MySQL Query Performance"** — FOSDEM 2023, **CC-BY-2.0-BE**
  (EXPLAIN plans, latency/throughput tables; different domain).
  https://archive.fosdem.org/2023/schedule/event/deep_dive_mysql_perf/

## Note on FOSDEM license tags

FOSDEM pages render both `by/2.0/be` (the talk video) and `by-sa/2.0/be` (the page
chrome); the speaker grant is attribution-class either way and does not block
publishing results. Attribute the speaker + FOSDEM when you publish.
