# Shared Layouts

## Application shell

- Source: `GUI/main_gui.py`
- Description: a fullscreen-capable `QMainWindow` containing five pages in a
  `QStackedWidget`. Navigation is signal-driven.

```python
class TriaGoApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TriaGO - Automated Medical Triage Kiosk")
        self.showMaximized()
        self.current_patient_info = {}
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.page_home = HomePage()
        self.page_registration = RegistrationPage()
        self.page_loading = LoadingPage()
        self.page_live_data = PlotPage()
        self.page_output = OutputPage()
        self.page_home.start_requested.connect(self.go_to_registration)
        self.page_registration.measurement_started.connect(
            self.handle_start_stabilization_phase
        )
        self.page_live_data.warmup_progress.connect(
            self.page_loading.update_ui_state
        )
        self.page_live_data.warmup_finished.connect(self.go_to_live_data_page)
        self.page_live_data.recording_finished.connect(
            self.handle_extraction_phase
        )
        self.page_output.home_requested.connect(self.reset_to_gatekeeper)
        self.stacked_widget.addWidget(self.page_home)
        self.stacked_widget.addWidget(self.page_registration)
        self.stacked_widget.addWidget(self.page_loading)
        self.stacked_widget.addWidget(self.page_live_data)
        self.stacked_widget.addWidget(self.page_output)
        self.stacked_widget.setCurrentIndex(0)
```

## Shared page frame

Every operational page uses:

```python
self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
self.setStyleSheet("background-color: #F6FFEC;")
main_layout = QVBoxLayout(self)
```

The current margins vary from 40–60 px horizontally and 20–50 px vertically.
For the target 1280×800 kiosk viewport, a consistent 28–32 px horizontal and
16–20 px vertical safe area is recommended.
