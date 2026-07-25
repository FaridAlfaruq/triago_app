# TriaGO 1280×800 Kiosk Design System

TriaGO is a touch-first automated medical triage kiosk displayed on a
Waveshare 10-inch HDMI screen in landscape orientation.

## Brand

- Preserve the real TriaGO logo from `asset/logo.png`.
- Use a calm clinical tone: light green canvas, navy information hierarchy,
  white measurement surfaces, explicit emergency colors.
- Avoid decorative effects that reduce medical readability.

## Layout

- Fixed design viewport: 1280×800.
- Fullscreen content; no operating-system panel or window chrome.
- Safe area: 32 px horizontal, 20 px vertical.
- Use flexible grid proportions rather than laptop-sized fixed widths.
- All essential controls must remain visible without scrolling.
- Minimum touch target: 52×52 px.

## Tokens

- Background: `#F6FFEC`
- Primary: `#214889`
- Primary dark: `#163264`
- Surface: `#FFFFFF`
- Border: `#C2D5BB`
- Muted text: `#556B85`
- Disabled: `#A0B09C`
- Selected: `#CEF9B6`
- Emergency: `#E74C3C`
- Warning: `#F39C12`
- Normal: `#2ECC71`
- Font: Segoe UI / Arial / sans-serif
- Radius: 10–12 px

## Page-specific constraints

### Registration

- Twelve bed buttons fit as a 6×2 grid inside 1216 px usable width.
- Recommended bed control: about 160×66 px with 16 px gaps.
- GCS 3–15 remains a single row of 13 touch targets, about 58×52 px.
- Header height should remain under 112 px.

### Recording

- Header uses one compact row.
- ECG and PPG plots receive equal vertical space.
- Plot labels must remain legible at 1280×800.

### Result

- Header status must never clip; allow status to shrink before title/logo.
- Give plots enough height for axes and labels.
- Parameter values must not overlap labels.
- Keep the return action visible at the bottom.
