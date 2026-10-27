"""
ESP32-CAM 사진 수신 -> 사람 감지(YOLO) -> 중복 제거 카운팅 서버

실행:  uvicorn server:app --host 0.0.0.0 --port 8000
엔드포인트:
  POST /upload      : JPEG 바이너리 수신 (ESP32-CAM이 호출)
  GET  /stats       : 누적 고유 인원 / 현재 프레임 인원
  GET  /latest.jpg  : 박스+ID가 그려진 최신 이미지
  POST /reset       : 누적 카운트 초기화
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ───────────── 설정 (환경에 맞게 튜닝) ─────────────
CONF_THRESH = 0.40      # 사람 감지 신뢰도
MIN_BOX_H = 60          # 이보다 작은 박스는 특징 추출이 부정확해서 중복 판정에서 제외
MIN_BOX_W = 25
MATCH_THRESH = 0.75     # 같은 사람으로 볼 유사도 (높을수록 엄격 = 사람이 더 많이 세어짐)
EMA_ALPHA = 0.2         # 매칭될 때 저장된 특징을 얼마나 갱신할지
GALLERY_TTL_SEC = 0     # 이 시간 이상 안 보이면 갤러리에서 삭제 (0 = 삭제 안 함)
SAVE_RAW = True         # 받은 원본 사진 저장 여부
# ────────────────────────────────────────────────────

SAVE_DIR = Path("received")
SAVE_DIR.mkdir(exist_ok=True)
LATEST_PATH = Path("latest.jpg")

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = None
backbone = None


def _ensure_models() -> None:
    """YOLO/백본 모델을 필요할 때만 로드해 초기화 에러를 줄인다."""
    global yolo, backbone

    if yolo is not None and backbone is not None:
        return

    try:
        if yolo is None:
            yolo = YOLO("yolov8n.pt")  # 처음 실행 시 자동 다운로드
    except Exception as exc:  # pragma: no cover - 동적 환경 의존
        raise RuntimeError(f"YOLO 모델 로드 실패: {exc}") from exc

    try:
        if backbone is None:
            try:
                _resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            except Exception:
                _resnet = torchvision.models.resnet18(weights=None)
            backbone = nn.Sequential(*list(_resnet.children())[:-1]).to(device).eval()
    except Exception as exc:  # pragma: no cover - 동적 환경 의존
        raise RuntimeError(f"백본 모델 로드 실패: {exc}") from exc


_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

app = FastAPI()
lock = threading.Lock()

# 갤러리: id -> {"feat": np.ndarray, "last_seen": float, "count": int}
gallery: dict[int, dict] = {}
next_id = 1
last_result: dict = {"in_frame": 0, "unique_total": 0}


# ───────────── 특징 추출 (CNN + 상/하의 색상 히스토그램) ─────────────
def _color_hist(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2HSV)
    parts = []
    for half in (hsv[:64], hsv[64:]):  # 상체 / 하체
        h = cv2.calcHist([half], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        parts.append(h / (np.linalg.norm(h) + 1e-6))
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-6)


def embed(crops: list[np.ndarray]) -> np.ndarray:
    """crop 리스트 -> (N, D) L2 정규화된 특징. 코사인 유사도 = 내적."""
    if not crops:
        return np.zeros((0, 1))

    _ensure_models()
    batch = []
    for c in crops:
        rgb = cv2.cvtColor(cv2.resize(c, (128, 256)), cv2.COLOR_BGR2RGB)
        batch.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    x = (torch.stack(batch).to(device) - _MEAN) / _STD
    with torch.no_grad():
        cnn = F.normalize(backbone(x).flatten(1), dim=1).cpu().numpy()
    hist = np.stack([_color_hist(c) for c in crops])
    w = np.sqrt(0.5)
    return np.concatenate([cnn * w, hist * w], axis=1)  # 전체 norm = 1


# ───────────── 매칭 ─────────────
def match_and_update(feats: np.ndarray):
    """같은 프레임의 서로 다른 박스는 서로 다른 사람 -> 헝가리안 1:1 매칭."""
    global next_id
    now = time.time()

    if GALLERY_TTL_SEC > 0:
        for pid in [p for p, g in gallery.items() if now - g["last_seen"] > GALLERY_TTL_SEC]:
            del gallery[pid]

    n = len(feats)
    ids = [None] * n
    best_sims = [0.0] * n
    gal_ids = list(gallery.keys())

    if n and gal_ids:
        G = np.stack([gallery[i]["feat"] for i in gal_ids])
        S = feats @ G.T
        rows, cols = linear_sum_assignment(-S)
        for r, c in zip(rows, cols):
            best_sims[r] = float(S[r, c])
            if S[r, c] >= MATCH_THRESH:
                pid = gal_ids[c]
                g = gallery[pid]
                f = (1 - EMA_ALPHA) * g["feat"] + EMA_ALPHA * feats[r]
                g["feat"] = f / (np.linalg.norm(f) + 1e-6)
                g["last_seen"] = now
                g["count"] += 1
                ids[r] = pid

    for i in range(n):
        if ids[i] is None:  # 새로운 사람
            gallery[next_id] = {"feat": feats[i], "last_seen": now, "count": 1}
            ids[i] = next_id
            next_id += 1
    return ids, best_sims


# ───────────── 한 장 처리 ─────────────
def process(data: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG 디코딩 실패 (전송 중 잘렸을 수 있음)")

    if SAVE_RAW:
        (SAVE_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.jpg").write_bytes(data)

    _ensure_models()
    h, w = img.shape[:2]
    res = yolo.predict(img, classes=[0], conf=CONF_THRESH, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy().astype(int)

    crops, kept, small = [], [], 0
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        if (y2 - y1) < MIN_BOX_H or (x2 - x1) < MIN_BOX_W:
            small += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 1)  # 회색 = 제외됨
            continue
        crops.append(img[y1:y2, x1:x2].copy())
        kept.append((x1, y1, x2, y2))

    feats = embed(crops) if crops else np.zeros((0, 1))
    with lock:
        ids, sims = match_and_update(feats) if crops else ([], [])
        unique_total = len(gallery)

    for (x1, y1, x2, y2), pid, s in zip(kept, ids, sims):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID {pid} ({s:.2f})", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f"in frame: {len(kept)}  unique total: {unique_total}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.imwrite(str(LATEST_PATH), img)

    result = {
        "in_frame": len(kept),
        "skipped_small": small,
        "ids": ids,
        "similarities": [round(s, 3) for s in sims],  # 임계값 튜닝용
        "unique_total": unique_total,
    }
    last_result.update(result)
    print(result)
    return result


# ───────────── API ─────────────
@app.post("/upload")
async def upload(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "빈 요청")
    try:
        return await run_in_threadpool(process, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/stats")
def stats():
    return {**last_result, "unique_total": len(gallery)}


@app.get("/latest.jpg")
def latest():
    if not LATEST_PATH.exists():
        raise HTTPException(404, "아직 받은 사진이 없음")
    return FileResponse(LATEST_PATH, media_type="image/jpeg")


@app.post("/reset")
def reset():
    global next_id
    with lock:
        gallery.clear()
        next_id = 1
    last_result.update({"in_frame": 0, "unique_total": 0})
    return {"ok": True}
