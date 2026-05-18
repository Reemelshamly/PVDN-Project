import json
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import subprocess
import uuid

app = FastAPI(title="PVDN Detection Service")

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

app.mount("/files", StaticFiles(directory=UPLOADS), name="files")


def resolve_model_path():
    candidates = []
    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        candidates.append(parent / "night" / "best_patch_model.pth")
        candidates.append(parent / "PVDN" / "night" / "best_patch_model.pth")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


@app.post("/api/detect")
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # save upload
    uid = uuid.uuid4().hex
    dest = UPLOADS / f"{uid}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # call detection runner (ports notebook inference). Uses run_detection.py
    runner = BASE / "run_detection.py"
    try:
        out_csv = UPLOADS / (dest.stem + "_log.csv")
        out_video = UPLOADS / (dest.stem + "_out.mp4")
        model_path = resolve_model_path()
        proc = subprocess.run([sys.executable, str(runner), str(dest), "--model", str(model_path), "--out-csv", str(out_csv), "--out-video", str(out_video)], capture_output=True, text=True, check=True)
        # runner prints JSON; return as-is
        try:
            parsed = json.loads(proc.stdout)
        except Exception:
            parsed = {"raw": proc.stdout}
        parsed["videoUrl"] = f"/files/{out_video.name}"
        parsed["csvUrl"] = f"/files/{out_csv.name}"
        parsed["sourceVideoUrl"] = f"/files/{dest.name}"
        return JSONResponse(content={"status": "ok", "output": parsed})
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"status": "error", "stderr": e.stderr, "stdout": e.stdout})
