# EUreqAI

**EUreqAI** is a developer-facing Python framework for assessing AI systems
against **Regulation (EU) 2024/1689 — the EU AI Act**.

> ⚠️ **Status: alpha, built in public.** Article mappings reference the final
> regulation as published in the Official Journal on 12 July 2024.
> This is *not legal advice*; use it as an engineering checklist alongside
> formal compliance work.

## Why now

| Date | Provisions becoming applicable |
| ---- | ------------------------------ |
| 2 Feb 2025 | Prohibited practices (Art. 5), AI literacy (Art. 4) |
| 2 Aug 2025 | GPAI model obligations (Art. 51–55), governance, penalties |
| 2 Aug 2026 | Most high-risk AI obligations (Annex III systems) |
| 2 Aug 2027 | Full applicability to legacy high-risk AI |

If you ship AI features for the EU market, the next 12 months are the window
to get your system, documentation and processes in shape.

## What's covered today

| Evaluator | Maps to | Status |
| --------- | ------- | ------ |
| `TransparencyEvaluator` | Art. 50 (disclosure to natural persons), Art. 13(3) | ✅ |
| `FairnessEvaluator` | Art. 10 (data governance, bias) | ✅ |
| `PrivacyEvaluator` | Art. 10(5) + GDPR Art. 32 | ✅ |
| `TechnicalRobustnessEvaluator` | Art. 15 (accuracy, robustness, cybersecurity) | ✅ |
| GPAI obligations | Art. 51–55 | 🛠 planned |
| Prohibited practices screening | Art. 5 | 🛠 planned |
| Annex IV technical documentation checklist | | 🛠 planned |

See the [issues board](https://github.com/EUreqAI/eureqai/issues) for the roadmap.

## Install

```bash
pip install -e .
# or, with dev tooling
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quickstart

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

Output (abbreviated):

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

This is a build-in-public project — issues, PRs and corrections from legal
and engineering reviewers are all welcome.

## License

[GNU Affero General Public License v3.0](LICENSE).
