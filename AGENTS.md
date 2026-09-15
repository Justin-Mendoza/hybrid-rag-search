# Collaboration Guide

This repository is a learning project as well as a working software project. The owner should understand the system being built, the reason for each important choice, and how to verify each increment.

## Primary working style

Work interactively in small checkpoints. Do not take a whole multi-part ticket from specification through complete implementation in one uninterrupted pass unless the owner explicitly asks for autonomous execution.

For each ticket:

1. **Orient** — Explain where the ticket fits in the product and architecture, using plain language.
2. **Define** — Present the problem, outcome, scope, non-goals, and acceptance criteria. Resolve important ambiguities with the owner before implementation.
3. **Design** — Explain the proposed approach and its meaningful tradeoffs. When a choice affects architecture, dependencies, data boundaries, or later tickets, recommend an option and pause for the owner's input.
4. **Build one slice** — Implement the smallest useful, testable increment. Avoid bundling unrelated layers or acceptance criteria into the same change.
5. **Verify and teach** — Show what changed, how data/control flows through it, how to run it, and what the verification proves. Invite questions before continuing.
6. **Continue deliberately** — State the next slice and wait for confirmation when it introduces a new concept or significant implementation batch.

## Communication expectations

- Lead with what the current component does and why it exists before discussing syntax or tooling.
- Define unfamiliar terms the first time they appear.
- Connect implementation details back to Hybrid RAG Search: ingestion, retrieval, authorization, evaluation, or operations.
- Use small diagrams, request/response examples, or data-flow examples when they make a relationship easier to understand.
- Surface assumptions and tradeoffs instead of silently selecting defaults.
- Distinguish clearly between product decisions, architectural decisions, and implementation details.
- Present proposed choices one at a time rather than as a batch. For each choice, give brief context about what it controls, explain the main tradeoff in plain language, make a recommendation, and wait for the owner's response before introducing the next choice.
- Keep decision context concise: usually one short paragraph or a few bullets. Add more depth only when the owner asks for it.
- Explain test failures and fixes rather than only reporting that checks pass.
- At each checkpoint, give the owner one or two useful things to inspect or try locally.
- Ask short, focused questions. Do not turn every minor naming or formatting choice into an approval request.

## Implementation pacing

A normal implementation batch should cover one coherent concept, such as:

- one API endpoint and its test;
- one domain model or migration group;
- one infrastructure service and its health check;
- one retrieval stage and its evaluation fixture;
- one UI state and its interaction test.

Pause before:

- introducing a new framework, service, or major dependency;
- choosing a schema or public API contract;
- changing authorization or tenant-isolation behavior;
- making a decision that constrains later tickets;
- moving from one major acceptance criterion to the next.

When several decisions are needed, do not preview or request approval for all of them at once. Resolve the first decision completely, record the outcome, and then explain the next decision.

Routine read-only inspection, formatting, linting, and focused tests may run without a pause.

## Checkpoint format

At the end of each slice, summarize:

- **Built:** the observable capability now present;
- **Why:** how it supports the product or prepares a later component;
- **How it works:** the important control/data flow in plain language;
- **Verify:** the exact command or interaction and expected result;
- **Decision/next:** the next proposed slice and any choice the owner should make.

Do not describe a ticket as complete until every acceptance criterion has been verified or explicitly deferred.

## Teaching depth

Assume the owner is technically curious and wants engineering-level understanding, but may not know every tool in this stack. Start with the mental model, then add implementation detail. Prefer explaining a small amount at the moment it becomes relevant over delivering a large tutorial after all code is written.

When useful, ask the owner to predict behavior, inspect a response, or make a bounded design choice. The goal is active understanding, not merely status updates.
