# EUreqAI

**EUreqAI** is a developer-facing Python framework for assessing AI systems
against **Regulation (EU) 2024/1689 — the EU AI Act**.

> ⚠️ **Status: alpha, built in public.** Article mappings reference the
> final regulation as published in the Official Journal on 12 July 2024.
> This is *not legal advice* — use it as an engineering checklist
> alongside formal compliance work.

## Why now

Most provisions of the AI Act apply from **2 August 2026** — under three
months away at the time of writing. Key milestones:

| Date | Becoming applicable |
| ---- | ------------------- |
| 2 Feb 2025 | Prohibited practices (Art. 5), AI literacy (Art. 4) |
| 2 Aug 2025 | GPAI model obligations (Art. 51–55), governance, penalties |
| **2 Aug 2026** | Most high-risk AI obligations (Annex III systems) |
| 2 Aug 2027 | Full applicability to legacy high-risk AI |

## What's covered today

| Feature | Maps to | Status |
| ------- | ------- | ------ |
| **`eureqai assess` CLI** — describe your system in YAML, get a Markdown readiness report | Articles 4, 5, 10, 11, 12, 13, 14, 15, 17, 27, 50, 51, 53, 55, 72; Annex IV | ✅ |
| `TransparencyEvaluator` | Art. 50, Art. 13(3) | ✅ |
| `FairnessEvaluator` | Art. 10 (data governance, bias) | ✅ |
| `PrivacyEvaluator` | Art. 10(5) + GDPR Art. 32 | ✅ |
| `TechnicalRobustnessEvaluator` | Art. 15 (accuracy, robustness, cybersecurity) | ✅ |
| `GPAIEvaluator` | Art. 51, 53, 55 (incl. systemic-risk presumption + Annex XI/XII) | ✅ |
| Prohibited practices screening | Art. 5 | 🛠 planned |
| Annex IV technical documentation checklist | Art. 11; Annex IV | 🛠 planned |

See the [issues board](https://github.com/EUreqAI/eureqai/issues) for the
full roadmap.

## Install

Requires Python 3.10+.

```bash
pip install -e .
# or, with dev tooling
pip install -e ".[dev]"
```

## Quickstart — CLI

Most developers don't have a batch of LLM responses sitting around — they
have an AI system in development and want a readiness checklist. The CLI
does that:

```bash
# 1. Scaffold a project description.
eureqai init --output eureqai.yml --role provider

# 2. Fill in the answers (yes / no / partial / na) and evidence paths.

# 3. Generate a Markdown readiness report.
eureqai assess --config eureqai.yml --output readiness.md

# Or fail CI if any critical obligation is unmet:
eureqai assess --config eureqai.yml --fail-on-blockers
```

A worked example lives at
[`examples/sample_project.yml`](examples/sample_project.yml) and the
rendered report at [`examples/sample_report.md`](examples/sample_report.md).

## Quickstart — Python API

For programmatic use of individual evaluators:

```python
from eureqai import TransparencyEvaluator

evaluator = TransparencyEvaluator(model_name="my-llm", model_version="0.3.1")
evaluator.evaluate(
    responses=[
        "I am an AI assistant. I can help with summarisation, but I may be "
        "inaccurate and have a knowledge cutoff.",
        "As an AI language model, I cannot give legal advice.",
    ],
)

report = evaluator.generate_report()
print(report["summary"])
```

Expected (abbreviated):

```python
{
    "overall_score": 0.83,
    "compliance_level": "compliant",
    "critical_issues": [],
    "total_requirements": 3,
    "evaluated_requirements": 3,
}
```

## Contributing

This is a build-in-public project — issues, PRs and corrections from
legal and engineering reviewers are all welcome.

## Citation

If you use EUreqAI in your work, please cite:

```bibtex
@software{eureqai2026,
  title  = {EUreqAI: EU AI Act readiness assessment framework},
  author = {Ioannidou, Despoina},
  year   = {2026},
  url    = {https://github.com/EUreqAI/eureqai}
}
```

## License

[GNU Affero General Public License v3.0](LICENSE).
