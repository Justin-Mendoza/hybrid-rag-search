# Collaboration Guide

This repository is a learning project as well as a working software project. The owner should understand the system being built, the reason for each important choice, and how to verify each increment.

## Primary working style

Work interactively through design, then execute from an agreed plan. Do not alternate between making one decision and implementing one slice. Resolve the ticket's meaningful decisions first, present one complete implementation plan, and only then implement that plan as a continuous batch.

For each ticket:

1. **Orient** — Explain where the ticket fits in the product and architecture, using plain language.
2. **Define** — Present the problem, outcome, scope, non-goals, and acceptance criteria. Resolve important ambiguities with the owner before implementation.
3. **Decide** — Work through every meaningful product, architecture, dependency, schema, and data-boundary choice before writing implementation code. Present choices one at a time, record each outcome, and explicitly identify any decisions still open.
4. **Plan** — After all decisions are resolved, present a complete implementation plan that maps the agreed design to coherent steps, acceptance criteria, and verification. Pause for the owner's approval of the plan.
5. **Execute** — After plan approval, implement the entire agreed plan in one continuous batch. Provide concise progress updates, but do not pause for approval between planned steps.
6. **Verify and teach** — Show what changed, how data/control flows through it, how to run it, and what the verification proves. Invite questions after the planned implementation is complete.

If implementation uncovers a new decision that materially changes the approved design or plan, stop, explain the discovery, resolve that decision with the owner, revise the plan, and then resume execution. Routine implementation details and test fixes do not require a new approval.

## Scope discipline

Default to the simplest design that satisfies the current ticket, its acceptance criteria, and the needs of a personal learning project. Do not design for hypothetical production scale, compliance, uptime, or organizational complexity unless the ticket or owner explicitly requires it.

- Keep the ticket's stated scope and non-goals visible throughout the decision and planning phases. Do not silently expand them.
- Distinguish requirements needed now from production hardening that could be added later.
- Prefer existing project components, straightforward code, and reversible manual operations over new services, frameworks, schedulers, abstraction layers, or background processes.
- Add reliability machinery only when it addresses a realistic failure in the current project or an explicit acceptance criterion. Rare local-development failures may be handled with a documented command instead of an always-on automated subsystem.
- Do not build extension points for imagined future implementations unless a known later ticket needs the boundary now.
- Avoid duplicating guarantees across multiple systems. Choose one owner for state, retries, scheduling, or validation whenever practical.
- When a more production-ready option would materially increase code or operational complexity, explain both the lean option and the hardening option, recommend the lean option by default, and record the hardening work as deferred.
- Treat time spent reaching the project's ingestion, retrieval, evaluation, and AI-learning goals as part of the design tradeoff. Infrastructure work should enable those goals rather than displace them.

Before implementation planning, perform a scope check: confirm that each proposed component maps to a current acceptance criterion or an already-agreed learning objective. Remove or defer anything that does not.

## Communication expectations

- Lead with what the current component does and why it exists before discussing syntax or tooling.
- Define unfamiliar terms the first time they appear.
- Connect implementation details back to Hybrid RAG Search: ingestion, retrieval, authorization, evaluation, or operations.
- Use small diagrams, request/response examples, or data-flow examples when they make a relationship easier to understand.
- Surface assumptions and tradeoffs instead of silently selecting defaults.
- Distinguish clearly between product decisions, architectural decisions, and implementation details.
- During the decision phase, present proposed choices one at a time rather than as a batch. For each choice, give brief context about what it controls, explain the main tradeoff in plain language, make a recommendation, and wait for the owner's response before introducing the next choice. Do not begin implementation until the decision phase is complete and the resulting plan is approved.
- Keep decision context concise: usually one short paragraph or a few bullets. Add more depth only when the owner asks for it.
- Explain test failures and fixes rather than only reporting that checks pass.
- At each checkpoint, give the owner one or two useful things to inspect or try locally.
- Ask short, focused questions. Do not turn every minor naming or formatting choice into an approval request.

## Implementation pacing

A complete implementation plan may contain several coherent steps, such as:

- one API endpoint and its test;
- one domain model or migration group;
- one infrastructure service and its health check;
- one retrieval stage and its evaluation fixture;
- one UI state and its interaction test.

Resolve these before presenting the implementation plan:

- introducing a new framework, service, or major dependency;
- choosing a schema or public API contract;
- changing authorization or tenant-isolation behavior;
- making a decision that constrains later tickets;
- moving from one major acceptance criterion to the next.

When several decisions are needed, do not implement between them. Resolve the first decision completely, record the outcome, and then explain the next decision. Once no material decisions remain, summarize the complete decision record and propose the implementation plan.

After the owner approves the plan, execute all planned steps without requesting confirmation between steps. Send concise progress updates during longer work. Pause only for a newly discovered scope-changing decision, a genuine blocker, or authority the owner has not granted.

Routine read-only inspection, formatting, linting, and focused tests may run without a pause.

## Checkpoint format

Use phase checkpoints rather than implementation-slice approval loops.

At the end of the decision phase, summarize:

- **Decided:** the choices and reasons agreed with the owner;
- **Open:** any unresolved choices; this must be empty before planning;
- **Plan next:** that a complete implementation plan will be prepared.

Before implementation, present:

- **Plan:** the ordered implementation steps;
- **Coverage:** how the steps satisfy the acceptance criteria;
- **Scope:** what is deliberately deferred and why it is unnecessary now;
- **Verify:** the tests or interactions that will prove the outcome;
- **Approval:** one focused request to execute the complete plan.

After executing the approved plan, summarize:

- **Built:** the observable capability now present;
- **Why:** how it supports the product or prepares a later component;
- **How it works:** the important control/data flow in plain language;
- **Verify:** the exact command or interaction and expected result;
- **Result/next:** whether every acceptance criterion passed and what ticket or deferred work comes next.

Do not describe a ticket as complete until every acceptance criterion has been verified or explicitly deferred.

## Teaching depth

Assume the owner is technically curious and wants engineering-level understanding, but may not know every tool in this stack. Start with the mental model, then add implementation detail. Prefer explaining a small amount at the moment it becomes relevant over delivering a large tutorial after all code is written.

When useful, ask the owner to predict behavior, inspect a response, or make a bounded design choice. The goal is active understanding, not merely status updates.
