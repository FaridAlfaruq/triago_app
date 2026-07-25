# Extractable Components

## AppShell

- Source: `GUI/main_gui.py`
- Category: layout
- Description: stacked five-screen kiosk shell.
- Extractable props: `currentPage`, `isFullscreen`.
- Hardcoded: page order, navigation flow, TriaGO window title.

## AppHeader

- Sources: `GUI/regist_page.py`, `GUI/plot_page.py`, `GUI/output_page.py`
- Category: layout
- Description: responsive title area with optional subtitle, status, progress,
  and brand logo.
- Extractable props: `title`, `subtitle`, `status`, `progress`.
- Hardcoded: logo asset, palette, typography.

## AnimatedProgressBar

- Source: `GUI/plot_page.py`
- Category: basic
- Description: animated pill progress meter.
- Extractable props: `value`.
- Hardcoded: TriaGO navy/sage colors.

## BedButton

- Source: `GUI/regist_page.py`
- Category: basic
- Description: large touch target for one of twelve beds.
- Extractable props: `label`, `selected`.
- Hardcoded: colors, border and number format.

## GCSButton

- Source: `GUI/regist_page.py`
- Category: basic
- Description: score selector colored by GCS severity.
- Extractable props: `score`, `selected`.
- Hardcoded: severity color thresholds.

## ParameterCard

- Source: `GUI/output_page.py::_create_param_card`
- Category: basic
- Description: compact medical measurement tile.
- Extractable props: `title`, `value`, `subvalue`.
- Hardcoded: surface, border, typography.

## TriageBadge

- Source: `GUI/output_page.py::update_triage_header`
- Category: basic
- Description: emergency classification color block and label.
- Extractable props: `status`.
- Hardcoded: red/yellow/green status mapping.
