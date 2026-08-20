# Design.md — AgentML (Streamlit Demo UI)

## 1. Design Philosophy
This is a technical/analytical tool, not a consumer product — prioritize clarity, information density, and trust signals (showing *reasoning*, not just results) over flashy visuals. The UI should feel like a lab notebook crossed with a dashboard: calm, legible, data-forward.

## 2. Color Palette

| Role | Color | Hex |
|---|---|---|
| Background (base) | Off-white | `#FAFAF8` |
| Background (panels/cards) | White | `#FFFFFF` |
| Primary text | Near-black | `#1A1A1A` |
| Secondary text | Slate gray | `#5C6470` |
| Primary accent (actions, active state) | Deep indigo | `#3B4C9B` |
| Success / best model | Muted green | `#2E7D5B` |
| Warning / checkpoint needed | Amber | `#C77D26` |
| Error / failed run | Muted red | `#B3413A` |
| Borders / dividers | Light gray | `#E4E6EA` |

Avoid saturated/neon colors — this tool's credibility comes from feeling careful and precise, not exciting.

## 3. Typography

| Use | Font | Notes |
|---|---|---|
| Headings | Inter (SemiBold) | Clean, technical, widely available |
| Body text | Inter (Regular) | Consistent with headings, no serif mixing |
| Code / data / metrics / logs | JetBrains Mono | Monospace for anything numeric or code-like — metric values, column names, file paths |

Type scale: H1 24px / H2 20px / H3 16px / Body 14px / Small 12px. Keep line-height generous (1.5) for the reasoning/log text blocks — these will be read carefully, not skimmed.

## 4. Layout Principles
- **Left-to-right pipeline visualization** at the top of the page showing the 7 agents as a horizontal flow, with the currently-active agent highlighted (indigo) and completed agents checked off (green). This directly supports the "transparency" goal from the PRD — the user should always see where they are in the pipeline.
- **Expandable sections per agent** below the pipeline view — each agent's output (profile stats, preprocessing log, detection reasoning, experiment leaderboard) is in its own collapsible card, collapsed by default except the currently active one.
- **Checkpoints render as a distinct card style** — amber left border, clear call-to-action buttons ("Confirm", "Override"), never blended in with regular progress cards.
- **Report view** is a clean, printable-feeling document pane — this is the deliverable, so it should look closer to a rendered Markdown/PDF document than a dashboard widget.

## 5. Components

| Component | Behavior |
|---|---|
| Upload zone | Drag-and-drop CSV, shows filename + row/col count immediately after profiling |
| Pipeline tracker | Horizontal stepper, 7 nodes, active/complete/pending states |
| Agent card | Collapsible, shows key output + a "Why?" toggle revealing the reasoning string |
| Checkpoint modal/card | Amber accent, shows the decision + confidence score, Confirm/Override buttons |
| Experiment leaderboard | Sortable table, best model row highlighted in green |
| Report pane | Rendered Markdown, download button (Markdown + PDF) |

## 6. Tone
Labels and copy should be plain and direct — "Detected task: Classification (confidence: 92%)" not "Our AI thinks this might be..." The tool's whole value proposition is precision and explainability, so the UI copy should never hedge or infantilize the output. Where confidence is genuinely low, say so plainly and route to the checkpoint rather than hiding uncertainty behind vague language.
