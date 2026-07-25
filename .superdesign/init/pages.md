# Page Dependency Trees

## Home

Entry: `GUI/home_page.py`

- `asset/logo.png`
- PyQt6 widgets / core / gui

## Registration: bed and GCS

Entry: `GUI/regist_page.py`

- `asset/logo.png`
- PyQt6 widgets / core / gui
- Python CSV and datetime

Rendered structure:

- Header: title + subtitle + logo
- White bed card: 12 bed buttons in a 6×2 grid
- GCS title + 13 buttons in one row
- Primary start button

## Loading

Entry: `GUI/loading_page.py`

- `processing_data/processing_data.py`
- `asset/logo.png`
- NumPy / pandas / PyQt6

Rendered structure:

- Centered logo
- Navy status card
- Animated progress bar
- Status text

## Live recording

Entry: `GUI/plot_page.py`

- `akuisisi_data/get_stm32.py`
- `processing_data/preprocessing_LiveData.py`
- `service/api_client.py`
- `asset/logo.png`
- PyQtGraph / NumPy / PyQt6

Rendered structure:

- Header: title + progress + logo
- ECG section label + plot card
- PPG section label + plot card

## Result

Entry: `GUI/output_page.py`

- `service/api_client.py`
- `asset/logo.png`
- PyQtGraph / NumPy / PyQt6

Rendered structure:

- Header: title/subtitle + triage badge + logo
- Top row: SHAP plot + ECG plot
- Bottom row: parameter cards + PPG plot
- Full-width return button

## Application shell

Entry: `GUI/main_gui.py`

- `GUI/home_page.py`
- `GUI/regist_page.py`
- `GUI/loading_page.py`
  - `processing_data/processing_data.py`
- `GUI/plot_page.py`
  - `akuisisi_data/get_stm32.py`
  - `processing_data/preprocessing_LiveData.py`
  - `service/api_client.py`
- `GUI/output_page.py`
  - `service/api_client.py`
