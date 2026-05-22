# Product Steering

## Purpose

This project provides a monitoring-first reference framework for running a simple PySpark pipeline on Kubernetes, where observability quality is the primary design goal.

## Target Users

- Platform operators who need fast, actionable visibility into pipeline health and failures.
- Data engineers who need lineage and run-level execution context for troubleshooting.
- Engineering leads who need a repeatable observability baseline for new pipelines.

## Core Value

- Detect pipeline failures quickly with consistent run context.
- Correlate metrics, traces, and lineage by run identity for faster root-cause analysis.
- Standardize the observability stack so onboarding is repeatable and low-friction.

## Product Boundaries

- In scope: local Kubernetes deployment, simple batch-style PySpark reference pipeline, monitoring validation workflows.
- Out of scope for v1: advanced pipeline business logic, multi-cluster orchestration, and external managed platform dependencies.

## Success Signals

- Failure and status transitions are visible to operators within minutes.
- Required observability components are present and connected by default.
- New simple pipelines can adopt the monitoring baseline with documented, repeatable steps.
