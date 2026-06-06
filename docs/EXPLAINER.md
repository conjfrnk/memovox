# What is memovox?

memovox turns videos into a **searchable, trustworthy memory** you can ask questions of.

Point it at a talk, lecture, meeting, or podcast and it works out what was *said* and
what was *shown*. Later you ask a plain-English question and get a plain-English answer.
Every sentence of that answer comes with a **receipt** — which video it came from, the
exact moment (with a link that jumps you straight there), who was speaking, and the
quote it's drawing from.

The one rule it never breaks: **if memovox can't point to where it learned something,
it won't claim it.**

## Why not just search the subtitles?

Most "chat with your video" tools only read the subtitle track. They chop the
transcript into chunks, find the chunks that look similar to your question, and hand
them to a chatbot. Two problems with that:

1. **They can make things up.** The chatbot might give a confident answer that isn't
   actually in the video — and you'd have no easy way to tell.
2. **They ignore the picture.** Slides, charts, code on screen, diagrams, captions —
   all the stuff you *see* in a video — get thrown away.

memovox fixes both: it reads the picture, not just the words, and never states anything
it can't trace back to a specific moment.

## How it works, in plain terms

Think of it as a careful research assistant who watches the whole video for you:

1. **Get the video.** From a local file, a transcript you already have, or a URL
   (URLs need an optional downloader and internet).
2. **Listen.** It transcribes the speech and works out who said what.
3. **Watch.** It picks out the most informative frames. With optional add-ons it also
   reads any on-screen text (slides, code, captions) and describes what's being shown.
4. **Stitch it together.** Spoken words and on-screen visuals that happen at the same
   time get bundled into small, timestamped *moments* — the basic unit it remembers.
5. **Pull out the facts — and check each one.** It extracts individual claims ("they
   recommend a 512-token chunk size") and confirms each one really is supported by its
   source moment before trusting it. Claims it can't back up are flagged and held out
   of the trusted memory — recorded, but never used to answer questions.
6. **Remember it.** Everything goes into a single searchable store, organized so it can
   later find things by meaning, by keyword, and by how facts relate to each other.
7. **Answer your questions.** When you ask, it finds the most relevant moments, writes
   an answer, and attaches a citation to every sentence.

In the code and docs each stage has a codename — Stentor (listen), Tessera (watch),
Escapement (stitch), Assay (check), Loom (remember), and Augur (answer) — but you never
need them to use memovox.

## What makes it different

- **Every answer is cited.** Each sentence ties back to a specific video, a specific
  timestamp, and whether the fact was spoken or shown on screen — with a link straight
  to that second.
- **It verifies before it remembers.** A claim is checked against its source before
  it's trusted, so the system never repeats things it can't back up.
- **It's free and runs on your machine.** No accounts, no API keys, nothing to
  download, no internet — a core install uses only what ships with Python. You can later
  plug in better speech recognition, smarter search, or a local language model for
  higher quality, with no change to how you use it.
- **It reads across videos, not just one.** Ingest many sources and it can summarize
  what they collectively say on a topic, surface where they **disagree**, and trace how
  an idea **changed over time**.
- **You can open the hood.** Everything lives in one ordinary SQLite database file plus
  plain-text Markdown summaries — so you can inspect it with everyday tools, not just
  memovox.

## Try it in 30 seconds

No accounts and nothing to download — just a sample transcript that ships with the repo.
From the repo root:

```bash
# 1. Load a sample transcript into a temporary store
PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ingest examples/scaling_laws.en.vtt --source-url https://youtu.be/SCALE123

# 2. Ask a question and get a cited answer
PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ask "what chunk size do they recommend?"
```

You'll get a one-line answer — **512 tokens** — with a `[1]` citation that links straight
to **0:24**, the exact second it's mentioned, tagged with the speaker (Dr. Lee).

## Where to go next

- **[README](../README.md)** — install options, the full command list, and the SDK,
  REST, and MCP interfaces.
- **[docs/DESIGN.md](DESIGN.md)** — how the pieces map to the code, and which model
  backends are pluggable.
- **[spec.md](../spec.md)** — the full design specification.
