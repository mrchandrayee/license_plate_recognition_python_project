from pathlib import Path
import re, uuid, shutil
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="AI License Plate Recognition", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE / "app" / "static")), name="static")
reader = None

# Common Indian registration format: SS##[A-Z]{1,3}####.
# Kept permissive enough to retain OCR candidates for human review.
PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")

def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False)
    return reader

def analyze_video(source: Path, destination: Path, sample_every: int = 5):
    ocr = get_reader()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")

    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    results, frame_index = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            # Upscaling helps OCR on small CCTV plates.
            scale = 1.5 if max(width, height) < 1000 else 1.0
            work = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame
            try:
                detections = ocr.readtext(work, detail=1, paragraph=False)
            except Exception:
                detections = []
            for bbox, raw, confidence in detections:
                candidate = normalize(raw)
                if len(candidate) < 7 or len(candidate) > 12:
                    continue
                # Accept likely registration strings; retain confidence for review.
                if not PLATE_RE.match(candidate):
                    continue
                pts = np.array(bbox, dtype=np.float32) / scale
                x1, y1 = pts.min(axis=0).astype(int)
                x2, y2 = pts.max(axis=0).astype(int)
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(width-1,x2), min(height-1,y2)
                key = candidate
                item = results.setdefault(key, {
                    "plate": candidate, "confidence": float(confidence),
                    "first_seen_seconds": round(frame_index / fps, 2),
                    "last_seen_seconds": round(frame_index / fps, 2),
                    "appearances": 0
                })
                item["appearances"] += 1
                item["last_seen_seconds"] = round(frame_index / fps, 2)
                item["confidence"] = max(item["confidence"], float(confidence))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(frame, f"{candidate} {confidence:.2f}",
                            (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,255,255), 2)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    return sorted(results.values(), key=lambda x: x["first_seen_seconds"]), frame_index, round(frame_index/fps,2)

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "app" / "static" / "index.html").read_text()

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Upload an MP4, AVI, MOV or MKV video.")
    job = uuid.uuid4().hex[:12]
    source = UPLOADS / f"{job}{suffix}"
    output = OUTPUTS / f"{job}_annotated.mp4"
    with source.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        plates, frame_count, duration = analyze_video(source, output)
        return {"job_id": job, "filename": file.filename, "frames_processed": frame_count,
                "duration_seconds": duration, "plates": plates,
                "annotated_video": f"/api/output/{job}"}
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(500, f"Video analysis failed: {exc}") from exc
    finally:
        source.unlink(missing_ok=True)

@app.get("/api/output/{job_id}")
def output(job_id: str):
    path = OUTPUTS / f"{job_id}_annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Output not found")
    return FileResponse(path, media_type="video/mp4", filename="annotated.mp4")
