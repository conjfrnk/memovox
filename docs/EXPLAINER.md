# What is memovox?

memovox turns videos into a **searchable, trustworthy memory** you can ask questions of.

Point it at a talk, lecture, meeting, interview, webinar, recorded class, or podcast —
including ones you recorded yourself, like a Zoom call or a family interview — and it works
out what was *said* and what was *shown*. It works on any recording with speech — the
subject matter doesn't matter. Later you ask a plain-English question and get a
plain-English answer. Every sentence of that answer comes with a **receipt** — which video
it came from, the exact moment (with a link that jumps you straight there), who was
speaking, and the quote it's drawing from.

The one rule it never breaks: **if memovox can't point to where it learned something,
it won't claim it.**

## Who is it for — and what do you need?

Anyone who lives in long recordings and wants answers they can trust: researchers,
journalists, students, teams reviewing meetings. One honest caveat: **memovox is a
command-line tool today.** You use it by typing a couple of short commands in a terminal
(the Terminal app on a Mac or Linux, PowerShell on Windows) — there is no clickable app
yet. If you've never opened a terminal, the copy-paste demo near the end shows exactly what
using memovox looks like — though getting it onto your computer in the first place (see the
README's Install section) is the step where a tech-comfortable friend helps most.

## Why not just search the subtitles?

Most "chat with your video" tools only read the subtitle track. They chop the
transcript into chunks, find the chunks that look similar to your question, and hand
them to a chatbot. Two problems with that:

1. **They can make things up.** The chatbot might give a confident answer that isn't
   actually in the video — and you'd have no easy way to tell.
2. **They ignore the picture.** Slides, charts, code on screen, diagrams, captions —
   all the stuff you *see* in a video — get thrown away.

memovox fixes both: it can read the picture too (with an optional add-on), not just the
words, and it never states anything it can't trace back to a specific moment.

## How it works, in plain terms

Think of it as a careful research assistant who watches the whole video for you:

1. **Get the video.** The simplest, fully-free way is to hand it a *transcript* you
   already have (most sites, including YouTube, let you download one). It can also take a
   local audio or video file, or a URL (URLs need an optional downloader and internet).
2. **Read the words.** If you gave it a transcript, it uses that; to turn raw audio into
   text it uses an optional speech-recognition add-on. Either way it tracks who said what —
   from the speaker labels already in the transcript, or, with another add-on, by telling
   the voices apart.
3. **Watch — only when there's a video.** It picks out the most informative frames. With
   optional add-ons it also reads any on-screen text (slides, code, captions) and describes
   what's being shown. Feeding it just a transcript skips this step, so reading the picture
   is an add-on, not part of the free default.
4. **Stitch it together.** Spoken words and any on-screen visuals that happen at the same
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
- **It's free, private, and runs on your machine.** Give it a transcript and everything
  happens offline, on your computer — no account, no API key, nothing uploaded. Your
  videos, transcripts, and answers never leave your machine. (Pulling a video from a URL is
  the one step that touches the internet, just to fetch that single file.) You can later
  plug in better speech recognition, smarter search, or a local language model for higher
  quality, with no change to how you use memovox. The model-based upgrades (better
  transcription, smarter search) download their files the first time you use them, then run
  locally; the local language model runs on a separate free program called
  [Ollama](https://ollama.com) that you set up once. The free path runs fine on a normal
  laptop; the optional AI upgrades also work there, just slower (a graphics card / GPU makes
  them fast).
- **It reads across videos, not just one.** Ingest many sources and it can summarize
  what they collectively say on a topic, surface where they **disagree**, and trace how
  an idea **changed over time**.
- **You can open the hood.** Everything lives in plain-text Markdown summaries you can open
  in any text editor, plus one ordinary database file — so nothing is locked inside memovox
  and you can inspect it with everyday tools.

## Try it (about 30 seconds, once you're set up)

**You'll type these into a terminal** (Terminal on Mac/Linux, PowerShell on Windows) —
remember, there's no clickable app yet. First you'll need the project's files on your
computer (clone or download the repo) and Python 3 installed — see the README's
[Install](../README.md#install) section. Then, from the repo's top folder, run the two
commands below against the sample transcript that ships with it (no accounts, nothing else
to download):

```bash
# 1. Load a sample transcript into a temporary store
PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ingest examples/scaling_laws.en.vtt --source-url https://youtu.be/SCALE123

# 2. Ask a question and get a cited answer
PYTHONPATH=src python -m memovox --store /tmp/mvdemo \
    ask "what chunk size do they recommend?"
```

You'll get an answer quoted straight from the talk. The sentence that answers your
question — *"the recommended chunk size is **512 tokens**"* — carries a `[1]` citation that
links to **0:24**, the exact second it's mentioned, tagged with the speaker (Dr. Lee). On
the free path the answer also quotes a few other relevant moments; add the optional language
model later to rewrite those quotes into a single, natural-language answer.

## Where to go next

- **[README](../README.md)** — install options, the full command list, and the SDK,
  REST, and MCP interfaces.
- **[docs/DESIGN.md](DESIGN.md)** — how the pieces map to the code, and which model
  backends are pluggable.
- **[spec.md](../spec.md)** — the full design specification.
