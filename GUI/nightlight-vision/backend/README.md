PVDN Detection Service

This folder contains a minimal FastAPI skeleton to run video detection on uploaded videos.

Quick start (venv):

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn detect_service:app --host 0.0.0.0 --port 8000
or 
python -m uvicorn detect_service:app --reload --port 8000

```

The endpoint `/detect` accepts `multipart/form-data` with a `file` field. The service saves
uploads to `uploads/` and executes `run_detection_stub.py <uploaded-file>` — replace the stub
with your real detection runner (adapt code from `night/08_blob_proposal_plus_tracking_improved.ipynb`).
