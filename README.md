# Backend Monitoring System

Lightweight toolkit for loading, preprocessing, training and running anomaly detection models on backend performance metrics.

**Quick Overview**
- **Purpose**: collect and process system metrics (CPU, memory, disk, network), train an autoencoder, and run realtime inference for anomaly detection.
- **Language**: Python

**Prerequisites**
- **Python 3.8+** installed.
- Project dependencies are listed in `requirements.txt`.

**Setup (Windows PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Project layout**
- `data/` : raw and processed CSV datasets
	- `data/raw/` : original CSV sources
	- `data/processed/` : outputs like `merged.csv`, `anomaly_results.csv`
- `src/` : main Python modules (`preprocess.py`, `train.py`, `inference.py`, etc.)
- `models/checkpoints/` : saved scaler and model checkpoints
- `tests/` : small test scripts

**Common commands**
- Preprocess (run tests / sanity check):
	- `python -m tests.test_preprocess`
- Train model:
	- `python -m src.train`
- Run inference (script):
	- `python -m src.inference`
- Run the realtime dashboard (Streamlit):
	- `streamlit run monitoring/realtime_dashboard.py`

**Notes**
- The preprocessing pipeline (`src/preprocess.py`) merges metric sources into `data/processed/merged.csv`.
- If you want the merged CSV without the `timestamp` column, remove or drop the `timestamp` column before saving in `src/preprocess.py` (e.g. `final.drop(columns=["timestamp"]).to_csv(...))`).

**Running tests**
- Run all tests with `pytest` (if installed):
	- `python -m pytest`
- Or run individual test modules:
	- `python -m tests.test_load`
	- `python -m tests.test_preprocess`

**Development tips**
- Use a virtual environment and ensure VS Code / Pylance is pointed at the same interpreter where you installed dependencies.
- If you see `Import "dotenv" could not be resolved`, install `python-dotenv` into the active environment and restart the language server.

**Next steps**
- Want me to update `src/preprocess.py` to automatically drop `timestamp` when writing `merged.csv`? I can apply that small change now.

---
Generated README to help contributors get started quickly.