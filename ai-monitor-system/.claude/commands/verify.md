---
description: Run an observability scenario against the live monitoring stack and report verdict.
---

Verify that the ai-monitor-system monitoring stack observed the pipeline
end-to-end by running a scenario's probes against live Prometheus / OTel /
Marquez / Alertmanager.

Argument: scenario name (without `.yaml`). Defaults to `success-baseline`.
Available scenarios live under `ai-monitor-system/scenarios/`.

Steps:
1. Spawn the `probe-runner` subagent with the scenario name. Do NOT run
   probes inline in the main session — the subagent exists specifically to
   keep verbose per-probe JSON out of this context window.
2. Wait for the subagent's single-paragraph verdict.
3. Relay the verdict to the user verbatim. Do NOT re-run probes to "verify
   the verifier". Do NOT auto-fix on failure unless the user asks.
4. If the verdict is ❌ or ⚠️, end your reply by asking the user whether
   they want you to investigate the suggested area.

Argument provided: $ARGUMENTS
