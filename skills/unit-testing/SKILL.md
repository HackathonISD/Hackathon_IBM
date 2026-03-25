---
name: unit-testing
description: Design, write, review, and harden unit, integration, and end-to-end tests with meaningful coverage and requirement-focused assertions.
compatibility: Strands AgentSkills
---

# Testing Skill

Use this skill when you need to create, revise, or assess unit tests, integration tests, or end-to-end tests.

## Objectives

- Map tests to user-visible requirements and important edge cases.
- Cover the system at the right levels: unit, integration, and end-to-end.
- Prefer deterministic, isolated tests over broad but flaky coverage.
- Catch regressions in business logic, validation, parsing, orchestration, and user flows.
- Keep tests readable enough that failures explain what broke.

## Workflow

1. Read the relevant specs, feature list, architecture notes, and source files before writing tests.
2. Partition the testing strategy into three levels:
  - unit tests for pure logic and small boundaries
  - integration tests for interactions between components, persistence layers, APIs, queues, or service boundaries
  - end-to-end tests for critical user journeys and system-level acceptance flows
3. For unit tests, identify the smallest meaningful boundaries: pure functions, service methods, validators, parsers, adapters, serializers, and permission checks.
4. For integration tests, cover contract edges, database access, API routes, workflow orchestration, and external boundary handling using realistic but controlled dependencies.
5. For end-to-end tests, cover the main user journeys that prove the requested product actually works from the outside.
6. At every level, cover:
  - nominal behavior
  - boundary conditions
  - invalid inputs
  - error propagation
  - any branching that affects business outcomes
7. Use fixtures and helpers only when they reduce duplication without hiding intent.
8. Mock external systems narrowly. Do not mock the unit under test, and do not over-mock internal logic that should be exercised directly.
9. Run the relevant test commands after changes and inspect failures before adding more tests.

## Quality Rules

- Test names should state the expected behavior, not just the function name.
- Assertions should be specific. Avoid assertions so broad that multiple bugs still pass.
- Avoid snapshot-style assertions unless the output is intentionally large and stable.
- Avoid sleeping, network calls, randomness, and clock dependence unless they are controlled.
- If code is hard to test because responsibilities are mixed, recommend or perform a small refactor that creates a clean seam for testing.
- Do not inflate coverage with shallow tests that ignore the actual requirements.
- End-to-end tests should stay focused on critical journeys, not duplicate every unit and integration scenario.
- Integration tests should verify real collaboration between components, not just re-run unit tests with more setup.

## Language Guidance

- For Python, prefer pytest-style tests with clear arrange/act/assert structure.
- Cover exceptions with explicit assertions on both the exception type and the critical message when relevant.
- When parameterization improves clarity, use it. When it obscures the scenario, write explicit tests instead.
- For API-heavy systems, favor integration tests around route handlers, service boundaries, and persistence rather than excessive controller mocking.
- For frontend or full-stack systems, use end-to-end tests only for key journeys such as authentication, CRUD flows, dashboards, and submission paths.

## Review Checklist

- Does each important requirement have coverage at the most appropriate level: unit, integration, or end-to-end?
- Do tests prove failure handling, not only success paths?
- Are mocks limited to true external boundaries?
- Can a future maintainer understand the failing behavior from the test name and assertions alone?
- Do the tests run quickly and deterministically for their level?