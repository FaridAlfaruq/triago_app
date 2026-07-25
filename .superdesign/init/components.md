# Shared UI Components

The application uses custom PyQt6 widgets rather than a third-party component
library. Most pages are monolithic. The reusable primitive is the animated
recording progress indicator.

## AnimatedProgressBar

- Source: `GUI/plot_page.py`
- Purpose: pill-shaped recording progress with TriaGO colors and animated value.
- Key API: `setValue(float)`, `animate_to(int)`.

```python
class AnimatedProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self.setFixedHeight(30)
        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def getValue(self):
        return self._value

    def setValue(self, v):
        self._value = max(0.0, min(100.0, v))
        self.update()

    value = pyqtProperty(float, fget=getValue, fset=setValue)

    def animate_to(self, target_value: int):
        self._animation.stop()
        self._animation.setStartValue(self._value)
        self._animation.setEndValue(float(target_value))
        self._animation.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        radius = rect.height() / 2
        track_path = QPainterPath()
        track_path.addRoundedRect(rect, radius, radius)
        painter.setPen(QPen(QColor("#C2D5BB"), 1.5))
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawPath(track_path)
        chunk_width = rect.width() * (self._value / 100.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#214889"))
        if chunk_width > rect.height():
            chunk_path = QPainterPath()
            chunk_path.addRoundedRect(
                QRectF(rect.x(), rect.y(), chunk_width, rect.height()),
                radius,
                radius,
            )
            painter.drawPath(chunk_path)
        elif chunk_width > 0:
            painter.drawEllipse(
                QRectF(rect.x(), rect.y(), rect.height(), rect.height())
            )
        painter.setPen(QColor("#FFFFFF") if self._value > 52 else QColor("#214889"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            f"{int(round(self._value))}%",
        )
```

## ParameterCard

- Source: `GUI/output_page.py::_create_param_card`
- Purpose: compact medical parameter tile used for temperature, heart rate,
  respiration, oxygen saturation, and blood pressure.

```python
def _create_param_card(self, grid_layout, title, default_val, row, col, colspan=1):
    card = QFrame()
    card.setStyleSheet(
        "QFrame { background-color: #F8FAF6; border: 1px solid #D5E5D0; "
        "border-radius: 8px; }"
    )
    vbox = QVBoxLayout(card)
    vbox.setContentsMargins(10, 6, 10, 6)
    vbox.setSpacing(1)
    lbl_title = QLabel(title)
    lbl_title.setStyleSheet(
        "font-size: 13px; font-weight: bold; color: #555555; "
        "border: none; background: transparent;"
    )
    lbl_val = QLabel(default_val)
    lbl_val.setStyleSheet(
        "font-size: 20px; font-weight: 900; color: #214889; "
        "border: none; background: transparent;"
    )
    lbl_sub = QLabel("")
    lbl_sub.setStyleSheet(
        "font-size: 11px; font-weight: 600; color: #778899; "
        "border: none; background: transparent;"
    )
    vbox.addWidget(lbl_title)
    vbox.addWidget(lbl_val)
    vbox.addWidget(lbl_sub)
    grid_layout.addWidget(card, row, col, 1, colspan)
    return lbl_val, lbl_sub
```
