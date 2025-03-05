# EUreqAI

**EUreqAI** is a Python framework for evaluating Large Language Models (LLMs) compliance with the EU AI Act requirements.

> **Disclaimer**: This project is a work in progress. Features and documentation are subject to change as development continues.

## Features

- Evaluation framework for LLM compliance
- Automated assessment of transparency, fairness, and technical requirements
- Reporting and recommendations
- Evidence-based compliance scoring
- Extensible architecture for custom requirements

## Installation

```python
pip install eureqai
from eureqai.evaluators import TransparencyEvaluator

# Initialize evaluator
evaluator = TransparencyEvaluator(
    model_name="your-model",
    model_version="1.0"
)

# Evaluate model responses
responses = [
    "I am an AI language model...",
    "As an artificial intelligence...",
]
results = evaluator.evaluate(responses=responses)

# Generate compliance report
report = evaluator.generate_report()
```

## License

This project is licensed under the GNU Affero General Public License v3.0

## Citation

If you use EUreqAI in your research, please cite:

```
@software{eureqai2024,
  title={EUreqAI: EU AI Act Compliance Framework for LLMs},
  author={[Despoina Ioannidou]},
  year={2024},
  url={https://github.com/EUreqAI/eureqai}
}
```
