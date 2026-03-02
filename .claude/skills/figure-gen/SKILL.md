---
name: figure-gen
description: Regenerate a manuscript figure, load and inspect every panel, proactively fix visual issues. Use when generating or regenerating any figure panel.
---

# Figure Generation & Inspection Workflow

After generating any manuscript figure panels, you MUST follow this protocol:

## Steps

1. **Run the figure generator** — call `generate()` from the appropriate figure module
2. **Load EVERY panel PNG** — use the Read tool to visually examine each panel image
3. **Check for these issues** (fix proactively, don't wait for user):
   - Empty panels or missing data
   - Overlapping labels, titles, or legends
   - Labels cut off at figure edges
   - Long crossing arrows from adjustText
   - Title/legend collisions
   - Missing axis labels or tick marks
   - Unreadable font sizes
   - Labels pushed outside axes bounds
4. **Fix any issues found** — edit the figure script and regenerate
5. **Re-inspect after fixes** — repeat until all panels are clean

## Key Rules

- NEVER claim "all panels verified" without actually loading and examining each image
- NEVER use `textcoords="offset points"` with large fixed offsets — use `adjustText` with `ensure_inside_axes=True`
- NEVER use simulated/synthetic data — all figures must use real datasets
- When using `fig.add_axes()` for marginal bars, call `fig.tight_layout(rect=...)` BEFORE `ax.get_position()`
- For adjustText: place labels at data coordinates with `ax.text()`, let `adjust_text()` nudge them — don't pass `x=`/`y=` parameters

## Figure Output Locations

- Main figures: `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/main/`
- Supplementary: `/Users/omarm/Documents/Research/projects/sc-trialdiff/manuscript/supp/`
- Python env: `/opt/anaconda3/bin/python`
