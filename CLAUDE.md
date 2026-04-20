# CLAUDE.md

## Purpose

This repository is a Streamlit dashboard. Your role is to improve the product's UI/UX, visual hierarchy, layout clarity, and interaction quality while preserving the underlying analysis, business logic, metrics, and data workflows unless explicitly asked to change them.

The desired design direction is:

- Calm, premium, restrained.
- Inspired by Apple’s product pages, settings panels, and dashboards.
- Elegant rather than trendy.
- Precise spacing, typography, contrast, and motion.
- Never generic startup UI.
- Never “AI slop”.

I am happy for you to change colors if it improves the overall design system.

---

## Core priorities

When working in this repo, optimize for the following in order:

1. Clarity of information.
2. Better hierarchy and scanability.
3. Higher visual quality.
4. Better spacing and alignment.
5. Better interaction flow.
6. Cleaner code organization.
7. Maintainability.

Do **not** prioritize decorative effects over readability or usability.

---

## Non-negotiables

- Preserve all existing calculations, filters, business rules, and data outputs unless explicitly instructed otherwise.
- Do not break working app behavior just to improve appearance.
- Avoid sweeping rewrites when targeted refactors will do.
- Prefer iterative improvement over total redesign unless asked for a redesign.
- Keep the app feeling like a serious analytical tool, not a marketing page.
- Every visual change must have a reason.

---

## Design direction

The target aesthetic should feel like:

- Apple-like restraint.
- Clean, quiet, and high trust.
- Spacious but not wasteful.
- Crisp typography.
- Soft surfaces, subtle depth, careful contrast.
- Clear grouping of controls, KPIs, charts, and tables.
- Premium dashboard rather than template dashboard.

Use these principles:

- Strong typography hierarchy.
- Fewer colors, used more intentionally.
- Larger margins and padding where they improve comprehension.
- Subtle borders and surfaces instead of loud cards.
- Minimal but polished interaction states.
- Consistent component patterns across the app.

---

## Explicitly avoid

Do not introduce any of the following unless I directly request them:

- Blue-purple gradients.
- Neon glows.
- Glassmorphism-heavy panels.
- Oversized rounded “blob” cards.
- Random accent colors.
- Template-looking KPI cards.
- Feature-grid SaaS aesthetics.
- Center-aligned dashboard content by default.
- Emoji as UI decoration.
- Empty decorative icons in colored circles.
- Visual noise disguised as polish.

If a choice feels “AI generated”, dial it back.

---

## Streamlit-specific guidance

Treat this as a real product UI built in Streamlit, not a quick notebook wrapper.

### Layout
- Prefer a clean top-down reading flow.
- Group related controls together.
- Reduce clutter in the sidebar; only keep high-value controls there.
- Use consistent width logic across pages and containers.
- Avoid cramped multi-column layouts unless the content clearly benefits.
- KPI sections should be compact, aligned, and easy to compare.
- Charts should have breathing room and visible titles/subtitles only where useful.
- Tables should feel deliberate, not default.

### Components
- Standardize:
  - metric cards
  - filter controls
  - tabs
  - expandable sections
  - chart containers
  - alert/info panels
  - table styling
- Reuse patterns instead of styling each section differently.
- Prefer fewer component styles with better consistency.

### Interaction
- Reduce friction in filters and page exploration.
- Make the “default view” feel intentional.
- Surface the most important insight first.
- Use progressive disclosure for secondary detail.
- Empty states, loading states, and validation states should feel designed.

### Visual system
- Build or refine a coherent design system for:
  - typography
  - spacing
  - color
  - borders
  - corner radius
  - shadows
  - chart styling
- Use a restrained palette with one primary accent and neutral support colors.
- If changing colors, choose something sophisticated and muted.

---

## Apple-inspired style rules

Use Apple as inspiration for quality and restraint.

### Typography
- Prioritize legibility and hierarchy over novelty.
- Use clean sans-serif typography with disciplined sizing.
- Avoid too many font sizes or weights.
- Titles should feel confident, not loud.
- Secondary text should remain readable, not washed out.

### Spacing
- Be generous and consistent.
- Increase whitespace where it improves grouping and comprehension.
- Align edges carefully.
- Use rhythm across sections so the interface feels composed.

### Surfaces
- Prefer soft neutral backgrounds.
- Use subtle elevation and low-contrast borders.
- Cards should feel quiet and precise, not chunky or ornamental.

### Motion
- Keep transitions subtle and purposeful.
- No flashy animation.
- Use motion only to support continuity or feedback.

### Data display
- Charts should feel editorial and calm.
- Reduce unnecessary legends, gridline weight, and label clutter.
- Use color intentionally to direct attention.
- Highlight one thing, not everything.

---

## UX objectives

Whenever you revise a page or component, improve these:

- Can a user understand the page in 3 seconds?
- Is the primary takeaway obvious?
- Are the key controls where a user expects them?
- Is the page easier to scan than before?
- Are labels concise and useful?
- Is visual emphasis placed only on important information?
- Does the layout help decision-making?

If the UI is prettier but not clearer, it is not better.

---

## Styling implementation preferences

When possible, improve styling through a clear system rather than scattered patchwork.

Prefer:
- shared spacing constants
- shared color tokens
- reusable card/container helpers
- consistent chart theme configuration
- standardized metric rendering
- consistent paddings/margins across sections

Avoid:
- repeated magic numbers
- slightly different card styles in different files
- one-off colors
- inconsistent plot styling
- ad hoc spacing per widget

---

## Chart and table guidance

For charts:

- Simplify visual noise.
- Use clean titles.
- Lighten gridlines.
- Avoid over-saturated palettes.
- Keep consistent typography and color usage across all charts.
- Use emphasis sparingly for the most important series or value.
- Format axes and labels for fast reading.

For tables/dataframes:

- Improve spacing and readability.
- Make columns easier to scan.
- Use formatting to guide attention, not decorate.
- Keep dense tables usable and calm.

---

## Sidebar guidance

The sidebar should feel useful, not crowded.

- Keep only core controls there.
- Group inputs logically.
- Use short section labels.
- Remove redundant helper text.
- Avoid making the sidebar the visual star of the app.

If filters are central to the workflow, make them feel structured and intentional.

---

## KPI and summary card guidance

KPI cards should:

- Be consistent in height and spacing.
- Use restrained visual treatment.
- Emphasize the number first.
- Keep labels compact.
- Use delta/trend indicators carefully.
- Never feel like colorful template tiles.

Aim for “quiet confidence”.

---

## Quality bar

A good change should make the app feel:

- more intentional
- more premium
- easier to use
- more trustworthy
- more cohesive

A bad change usually makes it feel:

- more generic
- more decorative
- more crowded
- more trend-chasing
- more obviously AI-generated

---

## Working style

Unless I ask otherwise:

- Start by explaining what you plan to improve.
- Then make the smallest set of high-value changes.
- After edits, summarize:
  - what changed
  - why it improved the UI/UX
  - any tradeoffs
  - any follow-up improvements worth considering

If the repo structure is unclear, inspect first rather than guessing.

---

## Default brief for UI tasks

When I ask for UI/UX improvement in this repo, assume I mean:

- refine layout
- improve visual hierarchy
- improve typography and spacing
- standardize components
- polish charts/tables
- reduce clutter
- make the app feel premium and modern
- preserve analytical seriousness
- avoid “AI slop”

Do not assume I want:
- new product features
- gimmicky animation
- landing-page aesthetics
- loud branding

---

## Definition of done

A UI/UX refinement task is complete only when:

- the app still works
- the layout is more coherent
- spacing is more consistent
- hierarchy is clearer
- components feel standardized
- charts/tables look more polished
- the interface feels more premium and less generic
- the result is restrained, not flashy
