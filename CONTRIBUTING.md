# Contributing to memovox

Thanks for your interest. memovox is solo-maintained — please open an issue to
discuss anything non-trivial before sending a pull request.

## Development

```bash
pip install -e ".[dev]"     # or just `pip install -e .` for the stdlib core
make test                   # full stdlib unittest suite (no pytest needed) — must pass
make lint                   # ruff, if installed
python -m eval.harness --assert-thresholds   # golden-corpus quality gates — must pass
```

Match the surrounding code style. Add tests for new behavior: the project is
TDD-disciplined and gates quality in CI, so a PR that changes behavior without a
test (or that trips an eval gate) won't merge.

## Licensing of contributions (please read)

memovox is **dual-licensed**: open source under **AGPL-3.0-or-later**, and under
separate **commercial licenses** offered by the copyright holder (see
[COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). For that model to work,
contributions must be licensable under **both**. By submitting a contribution you
agree to the following:

1. **Developer Certificate of Origin (DCO).** You certify the
   [DCO](https://developercertificate.org/) — you wrote the contribution, or have
   the right to submit it under these terms. Sign your commits with `git commit -s`
   (this adds a `Signed-off-by:` line).
2. **Inbound license + relicensing grant (lightweight CLA).** You license your
   contribution under AGPL-3.0-or-later, **and** you grant Connor (the project's
   copyright holder) a perpetual, worldwide, non-exclusive, royalty-free,
   irrevocable license to use, modify, and **sublicense your contribution under
   other terms, including commercial/proprietary licenses.** This lets the project
   keep offering commercial licenses without tracking down every contributor.

You keep the copyright in your contribution. This is not legal advice; for a
substantial contribution we may ask you to sign a standalone CLA.

## Reporting

- **Bugs:** open a GitHub issue with a minimal reproduction.
- **Security:** see [SECURITY.md](SECURITY.md) — report privately, not in a public issue.
