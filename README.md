# EUreqAI

**EUreqAI** is a developer-facing Python framework for assessing AI systems
against **Regulation (EU) 2024/1689 — the EU AI Act**.

> ⚠️ **Status: alpha, built in public.** Article mappings reference the final
> regulation as published in the Official Journal on 12 July 2024.
> This is *not legal advice*; use it as an engineering checklist alongside
> formal compliance work.

## What's covered today

| Evaluator | Maps to | Status |
| --------- | ------- | ------ |
| `TransparencyEvaluator` | Art. 50 (disclosure to natural persons), Art. 13(3) | ✅ |
| `FairnessEvaluator` | Art. 10 (data governance, bias) | ✅ |
| `PrivacyEvaluator` | Art. 10(5) + GDPR Art. 32 | ✅ |
| `TechnicalRobustnessEvaluator` | Art. 15 (accuracy, robustness, cybersecurity) | ✅ |
| `GPAIEvaluator` | Art. 51, 53, 55 (incl. systemic-risk presumption + Annex XI/XII) | ✅ |
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
@software{eureqai2024,
  title={EUreqAI: EU AI Act Compliance Framework for LLMs},
  author={Despoina Ioannidou},
  year={2024},
  url={https://github.com/EUreqAI/eureqai}
}
```
