<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Modified principles:
  - [PRINCIPLE_1_NAME] -> I. Code Quality by Design
  - [PRINCIPLE_2_NAME] -> II. Testing as Delivery Evidence
  - [PRINCIPLE_3_NAME] -> III. Consistent User Experience
  - [PRINCIPLE_4_NAME] -> IV. Performance Budgets are Requirements
  - [PRINCIPLE_5_NAME] -> V. Operational Simplicity and Maintainability
- Added sections:
  - Engineering Standards & Quality Gates
  - Technical Decision and Implementation Governance
- Removed sections:
  - None
- Templates requiring updates:
  - ✅ .specify/templates/tasks-template.md
  - ✅ .specify/templates/plan-template.md
  - ✅ .specify/templates/spec-template.md
  - ✅ .specify/templates/constitution-template.md
- Follow-up TODOs:
  - None
-->
# Always Try to Learn Constitution

## Core Principles

### I. Code Quality by Design
All production code MUST meet baseline quality standards: clear structure, bounded
module responsibilities, explicit error handling, lint/format compliance, and reviewable
readability. Pull requests MUST include rationale for non-obvious implementation choices
and avoid speculative complexity. Quality issues discovered during review or CI MUST be
resolved before merge, or explicitly accepted as time-bound debt with an owner and due date.

Rationale: Clean and understandable code reduces defects, accelerates onboarding, and keeps
future changes low-risk.

### II. Testing as Delivery Evidence
Every behavior change MUST be backed by automated tests at the appropriate level (unit,
integration, or contract). Bug fixes MUST include a regression test that fails before the fix
and passes after it. Test plans in specs and tasks MUST define what is validated, where it is
validated, and what constitutes a pass/fail outcome.

Rationale: Testing is the primary evidence that functionality works, remains stable, and
continues to satisfy user requirements over time.

### III. Consistent User Experience
User-facing flows MUST follow consistent interaction patterns, naming, response states, and
error messaging across the product. New features MUST document expected behavior for loading,
empty, success, and failure states, and MUST not introduce avoidable UX surprises.
Accessibility and clarity are required quality characteristics, not optional polish.

Rationale: Consistency increases trust, shortens learning time, and lowers support burden.

### IV. Performance Budgets are Requirements
Each feature MUST define measurable performance targets in planning (for example p95 latency,
throughput, startup time, or memory budget) and MUST validate them before release when the
feature impacts critical paths. Changes that risk violating budgets require explicit tradeoff
documentation and mitigation steps.

Rationale: Performance is a user-facing quality attribute and must be designed and verified
like functional behavior.

### V. Operational Simplicity and Maintainability
Implementation choices MUST prefer simplicity, observability, and debuggability over novelty.
Architectural or dependency complexity is acceptable only when justified by clear user value,
measurable reliability gains, or required scale constraints. Teams MUST retain the ability to
operate, diagnose, and evolve systems without specialist-only knowledge.

Rationale: Sustainable systems reduce delivery risk and improve long-term team velocity.

## Engineering Standards & Quality Gates

- Plans MUST include explicit code quality, test, UX, and performance checkpoints.
- Specifications MUST define acceptance criteria that are testable and user-observable.
- Tasks MUST include concrete validation work, including automated tests and performance checks
  when relevant.
- CI gates MUST include lint/format checks and test execution for impacted areas.
- Any temporary exception MUST document owner, expiry date, and rollback/remediation plan.

## Technical Decision and Implementation Governance

- Technical decisions MUST be traceable to user outcomes, product constraints, or operational
  needs; preference alone is insufficient.
- When multiple options exist, teams MUST record alternatives considered, decision criteria,
  and why the chosen option best satisfies this constitution.
- Implementation choices that weaken quality, testability, UX consistency, or performance MUST
  be escalated for explicit approval before merge.
- Architecture and dependency additions MUST include migration/rollback considerations and
  expected maintenance cost.
- Decision records MAY be lightweight but MUST be durable enough for future contributors to
  understand and challenge assumptions.

## Governance

This constitution is the highest-priority engineering policy for this repository. All plans,
specifications, tasks, code reviews, and release approvals MUST verify alignment.

Amendment process:
1. Propose a change with rationale, scope, and migration impact.
2. Review the change against current templates and update affected templates in the same change.
3. Obtain explicit maintainer approval before adoption.
4. Record the version bump and amendment date.

Versioning policy for this constitution:
- MAJOR: Incompatible governance or principle redefinition/removal.
- MINOR: New principle/section or materially expanded requirements.
- PATCH: Wording clarifications that do not change meaning.

Compliance review expectations:
- Every feature plan MUST pass a constitution check before design and before implementation.
- Every pull request MUST document how tests, UX behavior, and performance expectations were
  validated, or explain why a category is not applicable.
- Non-compliance MUST be tracked with owner, timeline, and follow-up remediation.

**Version**: 1.0.0 | **Ratified**: 2026-04-15 | **Last Amended**: 2026-04-15
