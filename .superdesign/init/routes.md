# Page / Route Map

The PyQt6 application uses a `QStackedWidget` rather than URL routing.

| Index | State | Component | Source |
|---:|---|---|---|
| 0 | Gatekeeper / home | `HomePage` | `GUI/home_page.py` |
| 1 | Bed and GCS registration | `RegistrationPage` | `GUI/regist_page.py` |
| 2 | Sensor warmup / signal processing | `LoadingPage` | `GUI/loading_page.py` |
| 3 | Live ECG and PPG recording | `PlotPage` | `GUI/plot_page.py` |
| 4 | Triage result | `OutputPage` | `GUI/output_page.py` |

## Navigation flow

```text
Home
  -> Registration
  -> Loading (sensor warmup)
  -> Live recording
  -> Loading (signal processing)
  -> Result
  -> Registration
```

The shell and all transitions are implemented in `GUI/main_gui.py`.
