"""Command-line entry point for ``eureqai``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from eureqai.checklist import CATALOGUE, items_for_role
from eureqai.config import load_config
from eureqai.report import assess, render_markdown


TEMPLATE_HEADER = """\
# AI Act readiness — project description for `eureqai assess`.
# See https://github.com/EUreqAI/eureqai for what each field means.
"""


def _build_template(role: str) -> str:
    items = items_for_role(role)
    lines: List[str] = [TEMPLATE_HEADER]
    lines.append("project:")
    lines.append("  name: my-ai-system")
    lines.append("  version: 0.1.0")
    lines.append("  organisation: Acme Health")
    lines.append('  intended_purpose: "Summarise referral letters for triage."')
    lines.append('  deployment_context: "EU hospitals; clinical decision support."')
    lines.append("")
    lines.append("classification:")
    lines.append(f"  role: {role}")
    lines.append("  risk_class: high_risk      # high_risk | gpai | limited_risk | minimal_risk | unknown")
    lines.append("  annex_iii_categories: []   # e.g. ['5(a)', '5(b)']")
    lines.append("")
    lines.append("documents:")
    lines.append("  # Free-form paths/URLs to evidence artefacts.")
    lines.append("  technical_documentation: docs/annex_iv.md")
    lines.append("  data_governance_policy: docs/data_governance.md")
    lines.append("  instructions_for_use: docs/ifu.md")
    lines.append("  risk_classification: docs/risk_classification.md")
    lines.append("  ai_literacy_programme: docs/ai_literacy.md")
    lines.append("  fria: docs/fria.md")
    lines.append("")
    lines.append("# Allowed answers: yes | no | partial | na")
    lines.append("answers:")
    for item in items:
        lines.append(f"  {item.id}: no   # {item.area} — {item.article}")
    lines.append("")
    lines.append("notes:")
    lines.append("  # Optional free-text per item.")
    lines.append(f"  {items[0].id}: \"\"")
    lines.append("")
    return "\n".join(lines)


def _init(args: argparse.Namespace) -> int:
    target = Path(args.output)
    if target.exists() and not args.force:
        print(
            f"error: {target} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 2
    target.write_text(_build_template(args.role), encoding="utf-8")
    print(f"Wrote template config to {target}")
    return 0


def _assess(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    assessment = assess(config)
    output = render_markdown(assessment)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        sys.stdout.write(output)
    if args.fail_on_blockers and assessment.overall_level == "not_ready":
        return 1
    return 0


def _catalogue(_: argparse.Namespace) -> int:
    for item in CATALOGUE:
        print(
            f"{item.id}\t{item.area}\t{item.article}\t{item.priority}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eureqai",
        description=(
            "Assess an AI system against Regulation (EU) 2024/1689 "
            "(the EU AI Act). Engineering aid only — not legal advice."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser(
        "init", help="Scaffold a project YAML for `eureqai assess`."
    )
    init_p.add_argument(
        "--output", "-o", default="eureqai.yml", help="Where to write the template."
    )
    init_p.add_argument(
        "--role",
        default="provider",
        choices=["provider", "deployer", "importer", "distributor", "gpai_provider"],
        help="Your role under the AI Act.",
    )
    init_p.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )
    init_p.set_defaults(func=_init)

    assess_p = sub.add_parser(
        "assess", help="Render a Markdown readiness report from a project YAML."
    )
    assess_p.add_argument(
        "--config", "-c", required=True, help="Path to the project YAML."
    )
    assess_p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write report here. Defaults to stdout.",
    )
    assess_p.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit non-zero if a critical item is marked 'no'.",
    )
    assess_p.set_defaults(func=_assess)

    cat_p = sub.add_parser(
        "catalogue", help="Print the underlying checklist catalogue."
    )
    cat_p.set_defaults(func=_catalogue)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
