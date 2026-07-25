# TriaGO Theme

## Compact token summary

### Colors

- Canvas / background: `#F6FFEC`
- Primary navy: `#214889`
- Primary pressed: `#163264` / `#0F2240`
- Border sage: `#C2D5BB`
- Soft card: `#F8FAF6`
- Card border: `#D5E5D0`
- Muted text: `#556B85`, `#555555`, `#778899`
- Disabled: `#A0B09C`
- Selection green: `#CEF9B6`
- Success: `#2ECC71` / `#34D980`
- Warning: `#F39C12` / `#F09C00`
- Emergency: `#E74C3C` / `#F12A2A`
- Surface: `#FFFFFF`

### Typography

- Family: `Segoe UI`, fallback `Arial`, sans-serif
- Display title: 32–40 px, weight 800–900
- Section title: 18–28 px, weight 700–800
- Body: 16–20 px, weight 400–600
- Compact labels: 11–13 px
- Medical values: 20 px, weight 900

### Shape and spacing

- Card radius: 12 px
- Compact tile radius: 8 px
- Pill radius: half component height
- Border: 1–2 px
- Current outer margins: 40–60 px horizontal, 20 px vertical
- Target 1280×800 outer margins: 28–32 px horizontal, 16–20 px vertical
- Target minimum touch control: 52×52 px

### Target viewport

- Waveshare HDMI 10-inch: `1280×800`, landscape, fullscreen kiosk
- No desktop panel or window title bar

## Raw source styles

```css
QWidget {
  background-color: #F6FFEC;
  color: #214889;
  font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame.card {
  border: 2px solid #C2D5BB;
  border-radius: 12px;
  background-color: #FFFFFF;
}
QPushButton.primary {
  background-color: #214889;
  color: #FFFFFF;
  font-size: 20px;
  font-weight: bold;
  border-radius: 10px;
  border: none;
}
QPushButton.disabled {
  background-color: #A0B09C;
  color: #FFFFFF;
}
```

No central stylesheet or theme provider currently exists; page-level
stylesheets in `GUI/*.py` are the source of truth.
