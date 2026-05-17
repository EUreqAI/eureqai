# Contributing

EUreqAI is built in public. Contributions from both engineers and legal /
compliance reviewers are very welcome — especially:

- Corrections to article mappings against **Regulation (EU) 2024/1689**.
- New evaluators for currently-planned items (e.g. prohibited practices,
  Annex IV technical-documentation checklist).
- Additional checklist items, role filters, or report formats.
- Better worked examples (`examples/`).

## Ground rules

This project is an **engineering aid, not legal advice**. PRs that
introduce text that could be mistaken for legal guidance will be asked to
add an explicit disclaimer. When in doubt, link to the exact article and
paragraph in the regulation rather than paraphrasing.

## Development setup

```bash
git clone https://github.com/EUreqAI/eureqai.git
cd eureqai
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the full check before opening a PR:

```bash
pytest -q
flake8 src tests
```

CI runs the same on Python 3.10, 3.11 and 3.12.

## Opening a PR

- Branch from `main`.
- Keep one feature or one fix per PR — easier to review, easier to revert.
- Update the README's "What's covered today" table if the change adds or
  removes a feature row.
- For new evaluators, follow the existing pattern: subclass
  `BaseEvaluator`, declare requirements in `_initialize_requirements`,
  return `EvaluationResult` from `evaluate()`, and include at least a
  smoke test.

## Reporting issues

Bugs, broken article references, or suggestions for new checklist items
all go in the [issue tracker](https://github.com/EUreqAI/eureqai/issues).
For security-relevant issues, please open a private security advisory on
GitHub instead of a public issue.
