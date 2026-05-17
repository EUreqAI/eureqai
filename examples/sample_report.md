# AI Act readiness report — ReferralTriageAssistant

_Generated 2026-05-16 18:35 UTC by EUreqAI._

## Project

- **Version**: 0.4.0
- **Organisation**: Acme Health
- **Intended purpose**: Prioritise incoming patient referral letters per hospital department, suggest a triage category for clinician review. Final decision is always taken by a human clinician.
- **Deployment context**: EU hospitals; embedded in the EHR's referral inbox. Outputs are advisory and never executed without clinician sign-off.
- **Role under the AI Act**: `provider`
- **Risk classification**: `high_risk`
- **Annex III categories**: 5(a)

## Summary

**Overall readiness**: 🟢 Ready (score 0.81 / 1.00)

| Area | Score | Status |
| ---- | ----- | ------ |
| Prohibited practices | 1.00 | 🟢 Ready |
| AI literacy | 0.50 | 🔴 Not ready |
| Risk classification | 1.00 | 🟢 Ready |
| Data governance | 0.75 | 🟡 Gaps to close |
| Technical documentation | 1.00 | 🟢 Ready |
| Record-keeping | 1.00 | 🟢 Ready |
| Transparency | 1.00 | 🟢 Ready |
| Human oversight | 1.00 | 🟢 Ready |
| Technical robustness | 0.50 | 🔴 Not ready |
| Governance | 0.00 | 🔴 Not ready |
| GPAI obligations | 1.00 | 🟢 Ready |
| Fundamental rights | 1.00 | 🟢 Ready |

## Detailed findings

### Prohibited practices — 🟢 Ready

- ✅ **[PROHIB-1]** (Article 5, critical) — Have you confirmed the system does NOT engage in any practice prohibited by Article 5 (subliminal manipulation, exploitation of vulnerabilities, social scoring, untargeted scraping of facial images, emotion recognition in workplace/education, biometric categorisation for protected attributes, real-time remote biometric ID in public spaces, predictive policing based solely on profiling)?

### AI literacy — 🔴 Not ready

- 🟡 **[LIT-1]** (Article 4, high) — Do staff who operate or are affected by the AI system receive AI-literacy training appropriate to their role (Article 4)?
  - _Evidence_: `docs/ai_literacy.md`
  - _Note_: Training programme exists for clinicians; admin staff onboarding is being drafted, target Q3.

### Risk classification — 🟢 Ready

- ✅ **[RISK-1]** (Articles 6, 50, 51; Annex III, critical) — Have you documented whether the system is high-risk under Article 6 / Annex III, GPAI under Article 51, limited-risk under Article 50, or minimal-risk?
  - _Evidence_: `docs/risk_classification.md`

### Data governance — 🟡 Gaps to close

- ✅ **[DATA-1]** (Article 10(2)–(3), high) — Do training, validation and testing datasets meet the quality criteria in Article 10(2)–(3): relevance, representativeness, freedom from errors, and bias examination?
  - _Evidence_: `docs/data_governance.md`
- 🟡 **[DATA-2]** (Article 10(2)(b); GDPR Article 5(1)(c), high) — Is data lineage documented and is collection minimised to what is strictly necessary for the intended purpose?
  - _Note_: Lineage is documented for ETL, but minimisation review for two secondary fields is outstanding.

### Technical documentation — 🟢 Ready

- ✅ **[DOC-1]** (Article 11; Annex IV, critical) — For a high-risk system, is technical documentation prepared in line with Annex IV (general description, design choices, datasets, validation, monitoring) before placing on the market?
  - _Evidence_: `docs/annex_iv.md`

### Record-keeping — 🟢 Ready

- ✅ **[LOG-1]** (Article 12, high) — Does the system automatically log events sufficient to ensure traceability over its lifecycle (Article 12)?

### Transparency — 🟢 Ready

- ✅ **[TRANS-1]** (Article 50, critical) — Are end users informed when interacting with an AI system, and is AI-generated/manipulated content marked as such where Article 50 requires it?
- ✅ **[TRANS-2]** (Article 13, high) — Do deployers receive instructions for use covering intended purpose, accuracy/robustness levels, foreseeable misuse and limitations (Article 13)?
  - _Evidence_: `docs/ifu.md`

### Human oversight — 🟢 Ready

- ✅ **[OVER-1]** (Article 14, critical) — Are human-oversight measures designed in: ability to monitor, interpret outputs, intervene and stop the system (Article 14)?

### Technical robustness — 🔴 Not ready

- 🟡 **[TECH-1]** (Article 15, critical) — Are accuracy, robustness and cybersecurity levels declared and tested, including resilience to adversarial inputs (Article 15)?
  - _Note_: Accuracy and robustness are declared; adversarial testing for prompt injection scheduled for the next sprint.

### Governance — 🔴 Not ready

- ❌ **[QMS-1]** (Articles 17, 72, high) — For high-risk: is a quality management system in place (Article 17), and a post-market monitoring plan (Article 72)?
  - _Note_: QMS work has not started — currently the biggest gap and the reason we cannot place the system on the market yet.

### GPAI obligations — 🟢 Ready

- ⚪ **[GPAI-1]** (Articles 51, 53, 55, critical) — If providing a GPAI model: technical documentation, copyright policy and a sufficiently detailed summary of training content (Article 53) — and systemic-risk evaluations under Article 55 where the model meets the Article 51 threshold?
  - _Note_: Not a GPAI provider; we deploy a third-party LLM and rely on the provider's GPAI documentation.

### Fundamental rights — 🟢 Ready

- ✅ **[FRIA-1]** (Article 27, high) — If you are a deployer covered by Article 27 (public bodies and certain banking/insurance use cases), have you carried out a fundamental rights impact assessment?
  - _Evidence_: `docs/fria.md`

---

_This report is an engineering aid, not legal advice. Validate with qualified counsel before relying on it for conformity decisions._
