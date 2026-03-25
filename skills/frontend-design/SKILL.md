---
name: frontend-design
description: Build professional frontend interfaces with consistent visual systems, no emoji-heavy styling, and color palettes inspired by reputable design references.
compatibility: Strands AgentSkills
---

# Frontend Design Skill

Use this skill when the task includes UI, pages, dashboards, or visual components.

## Goals

- Produce professional-looking interfaces suitable for production demos.
- Keep a consistent visual language across all pages and components.
- Avoid emoji-heavy UI and decorative gimmicks in product interfaces.
- Choose a deliberate color palette based on reputable design references.

## Required design principles

- Define a color system up front with named tokens (primary, secondary, accent, background, surface, text, muted, success, warning, error).
- Keep color usage consistent across states and components.
- Ensure readable contrast for text, controls, and status indicators.
- Use consistent spacing, border radius, shadows, and typography scale.
- Keep layouts responsive for desktop and mobile.

## Palette research workflow

1. Use `tavily_search` to find professional color palette and design-system references.
2. Prioritize reputable sources such as:
   - Material Design guidance
   - Atlassian Design System
   - Shopify Polaris
   - Adobe Spectrum
   - Tailwind color documentation and curated palette resources
3. Use the `browser` tool to inspect shortlisted references.
4. Select one palette direction and document why it matches the product domain.
5. Apply the palette through reusable variables/tokens instead of hardcoding colors everywhere.

## Styling constraints

- Do not use emojis in UI labels, titles, buttons, badges, or navigation.
- Do not mix unrelated color families without a documented rationale.
- Do not apply random per-component colors that break visual consistency.
- Avoid low-contrast text on colored backgrounds.
- Avoid over-animating the interface.

## Implementation checklist

- Define root color tokens or theme variables.
- Define typography tokens (font sizes, weights, line heights).
- Define spacing and radius scale tokens.
- Verify hover/focus/active/disabled states with the same visual system.
- Verify responsive behavior on small and large screens.
- Verify empty, loading, and error states are styled consistently.

## Review checklist

- Is the interface professional and coherent without emoji styling?
- Is one palette direction consistently applied everywhere?
- Are text and controls readable with sufficient contrast?
- Are reusable style tokens used instead of one-off values?
- Does the UI look intentional on both desktop and mobile?