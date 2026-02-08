\# LeastRisk Selector



LeastRisk Selector is a standalone Streamlit application that verifies, scores, and automatically selects the least-risk candidate output from multiple alternatives using explicit, auditable rules.



\## Features

\- Rule-based verification (hard and soft constraints)

\- Deterministic risk scoring

\- Automatic least-risk selection

\- Visual explanations (heatmaps, stacked risk bars, drill-down tables)

\- Downloadable audit record (JSON)



\## Requirements

\- Python 3.10+

\- Streamlit



\## Install \& Run



```bash

python -m venv .venv

\# Windows

.venv\\Scripts\\activate

\# macOS/Linux

\# source .venv/bin/activate



pip install -r requirements.txt

streamlit run app.py



