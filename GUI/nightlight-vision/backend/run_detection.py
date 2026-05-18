import sys
from pathlib import Path
import shutil
import json
import traceback
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from scipy.ndimage import label, find_objects, binary_dilation
from collections import deque
import pandas as pd


# Config (kept minimal; mirrors notebook defaults)
IMAGE_SIZE = 128
BATCH_SIZE = 64
BLOB_K = 0.65
BLOB_W = 21
PADDING = 8
DEV_THRESH = 0.025
NMS_DISTANCE = 5
SMALL_SCALE = (640, 480)

ROI_Y1_RATIO = 0.45
ROI_Y2_RATIO = 0.95
ROI_X1_RATIO = 0.18
ROI_X2_RATIO = 0.82

MIN_PROPOSAL_AREA = 70
MAX_PROPOSAL_AREA = 4500
MIN_PROPOSAL_WIDTH = 6
MAX_PROPOSAL_WIDTH = 220
MIN_PROPOSAL_HEIGHT = 4
MAX_PROPOSAL_HEIGHT = 120
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 8.0
MIN_MEAN_INTENSITY = 95
MIN_MAX_INTENSITY = 155
MIN_BRIGHT_PIXEL_RATIO = 0.015
PROPOSAL_QUALITY_THRESHOLD = 0.18

PATCH_THRESHOLD = 0.97
TOP_K_DETECTIONS = 5
MIN_FINAL_SCORE = 0.45

MAX_TRACK_DISTANCE = 85
MAX_MISSED_FRAMES = 6
MIN_TRACK_HITS_FOR_WARNING = 5
MIN_CONSECUTIVE_HITS_FOR_WARNING = 3
TRACK_CONF_THRESHOLD = 0.70
TRACK_SCORE_WINDOW = 8
MIN_WARNING_TRACKS = 1
REQUIRE_TRACK_MOTION = False
MIN_TRACK_MOTION_PIXELS = 8


def resolve_model_path(explicit_path=None):
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidates.append(parent / "night" / "best_patch_model.pth")
        candidates.append(parent / "PVDN" / "night" / "best_patch_model.pth")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path(explicit_path) if explicit_path else candidates[0]


def load_model(model_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


patch_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def integral_image(img):
    return cv2.integral(img)


def box_integral(ii, box):
    x1, y1, x2, y2 = box
    return ii[y1, x1] + ii[y2, x2] - ii[y1, x2] - ii[y2, x1]


def box_area(box):
    x1, y1, x2, y2 = box
    return max((x2 - x1) * (y2 - y1), 1)


def box_mean(ii, box):
    return box_integral(ii, box) / box_area(box)


def dynamic_threshold(img, k=0.4, window=19, eps=1e-6):
    img_float = img.astype(np.float32) / 255.0
    mean = cv2.blur(img_float, (window, window))
    mean_sq = cv2.blur(img_float ** 2, (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, eps))
    thresh = mean + k * std
    binary = img_float > thresh
    return binary.astype(np.uint8)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_boxes(boxes, scores, iou_threshold=0.30, top_k=80):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0 and len(keep) < top_k:
        idx = order[0]
        keep.append(idx)
        remaining = []
        for j in order[1:]:
            if box_iou(boxes[idx], boxes[j]) <= iou_threshold:
                remaining.append(j)
        order = np.array(remaining)
    return [boxes[i] for i in keep]


def compute_proposal_quality(frame_gray, box):
    h, w = frame_gray.shape
    x1, y1, x2, y2 = box
    patch = frame_gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    mean_intensity = float(np.mean(patch))
    max_intensity = float(np.max(patch))
    std_intensity = float(np.std(patch))
    bright_ratio = float(np.mean(patch >= 190))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    center_distance = abs(cx - (w / 2)) / (w / 2)
    center_score = max(0.0, 1.0 - center_distance)
    lower_score = np.clip((cy / h - ROI_Y1_RATIO) / max(1e-6, ROI_Y2_RATIO - ROI_Y1_RATIO), 0, 1)
    brightness_score = np.clip((mean_intensity - 80) / 120, 0, 1)
    max_score = np.clip((max_intensity - 130) / 125, 0, 1)
    contrast_score = np.clip(std_intensity / 60, 0, 1)
    bright_ratio_score = np.clip(bright_ratio / 0.08, 0, 1)
    quality = (0.25 * brightness_score + 0.20 * max_score + 0.20 * contrast_score + 0.20 * bright_ratio_score + 0.10 * center_score + 0.05 * lower_score)
    return float(quality)


def passes_proposal_filters(frame_gray, box):
    h, w = frame_gray.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh
    if area < MIN_PROPOSAL_AREA or area > MAX_PROPOSAL_AREA:
        return False
    if bw < MIN_PROPOSAL_WIDTH or bw > MAX_PROPOSAL_WIDTH:
        return False
    if bh < MIN_PROPOSAL_HEIGHT or bh > MAX_PROPOSAL_HEIGHT:
        return False
    aspect_ratio = bw / max(1, bh)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < w * ROI_X1_RATIO or cx > w * ROI_X2_RATIO:
        return False
    if cy < h * ROI_Y1_RATIO or cy > h * ROI_Y2_RATIO:
        return False
    patch = frame_gray[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    mean_intensity = float(np.mean(patch))
    max_intensity = float(np.max(patch))
    bright_ratio = float(np.mean(patch >= 190))
    if mean_intensity < MIN_MEAN_INTENSITY:
        return False
    if max_intensity < MIN_MAX_INTENSITY:
        return False
    if bright_ratio < MIN_BRIGHT_PIXEL_RATIO:
        return False
    quality = compute_proposal_quality(frame_gray, box)
    if quality < PROPOSAL_QUALITY_THRESHOLD:
        return False
    return True


def generate_blob_proposals(frame_gray):
    h_original, w_original = frame_gray.shape
    x1_roi = int(w_original * ROI_X1_RATIO)
    x2_roi = int(w_original * ROI_X2_RATIO)
    y1_roi = int(h_original * ROI_Y1_RATIO)
    y2_roi = int(h_original * ROI_Y2_RATIO)
    roi = frame_gray[y1_roi:y2_roi, x1_roi:x2_roi]
    h_roi, w_roi = roi.shape
    resized = cv2.resize(roi, SMALL_SCALE, interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    cv2.normalize(resized, resized, 0, 255, cv2.NORM_MINMAX)
    resized = resized.astype(np.uint8)
    blurred = cv2.GaussianBlur(resized, (5, 3), 2)
    binary = dynamic_threshold(blurred, k=BLOB_K, window=BLOB_W)
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    if NMS_DISTANCE > 1:
        structure = np.ones((NMS_DISTANCE, NMS_DISTANCE), dtype=np.uint8)
        binary = binary_dilation(binary, structure=structure).astype(np.uint8)
    labeled = label(binary)[0]
    objects = find_objects(labeled)
    proposals_small = []
    img_float = blurred.astype(np.float32) / 255.0
    ii = integral_image(img_float)
    for obj in objects:
        if obj is None:
            continue
        y_slice, x_slice = obj
        sx1 = x_slice.start
        sx2 = x_slice.stop
        sy1 = y_slice.start
        sy2 = y_slice.stop
        sw = sx2 - sx1
        sh = sy2 - sy1
        proposal_size = sw * sh
        if proposal_size <= 8:
            continue
        if proposal_size >= 1800:
            continue
        if sw <= 4 or sh <= 2:
            continue
        small_box = [sx1, sy1, sx2, sy2]
        patch = img_float[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            continue
        mean_val = box_mean(ii, small_box)
        deviation = np.abs(patch - mean_val).sum() / proposal_size
        if deviation > DEV_THRESH:
            proposals_small.append(small_box)
    scale_x = w_roi / SMALL_SCALE[0]
    scale_y = h_roi / SMALL_SCALE[1]
    proposals = []
    scores = []
    for small_box in proposals_small:
        sx1, sy1, sx2, sy2 = small_box
        bx1 = int(sx1 * scale_x) + x1_roi
        by1 = int(sy1 * scale_y) + y1_roi
        bx2 = int(sx2 * scale_x) + x1_roi
        by2 = int(sy2 * scale_y) + y1_roi
        bx1 = max(bx1 - PADDING, 0)
        by1 = max(by1 - PADDING, 0)
        bx2 = min(bx2 + PADDING, w_original)
        by2 = min(by2 + PADDING, h_original)
        box = [bx1, by1, bx2, by2]
        if not passes_proposal_filters(frame_gray, box):
            continue
        quality = compute_proposal_quality(frame_gray, box)
        proposals.append(box)
        scores.append(quality)
    proposals = nms_boxes(proposals, scores, iou_threshold=0.25, top_k=60)
    return proposals


def crop_proposal_context(frame_gray, box, output_size=128, support_multiplier=3):
    h, w = frame_gray.shape
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw // 2
    cy = y1 + bh // 2
    radius = max(bw, bh, 32) * support_multiplier
    nx1 = max(int(cx - radius), 0)
    ny1 = max(int(cy - radius), 0)
    nx2 = min(int(cx + radius), w)
    ny2 = min(int(cy + radius), h)
    crop = frame_gray[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return crop


def classify_proposals(frame_gray, proposals, model, device):
    if len(proposals) == 0:
        return []
    tensors = []
    valid_boxes = []
    qualities = []
    for box in proposals:
        crop = crop_proposal_context(frame_gray, box, output_size=IMAGE_SIZE, support_multiplier=2.2)
        if crop is None:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = patch_transform(pil_img)
        quality = compute_proposal_quality(frame_gray, box)
        tensors.append(tensor)
        valid_boxes.append(box)
        qualities.append(quality)
    if len(tensors) == 0:
        return []
    batch = torch.stack(tensors).to(device)
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(batch), BATCH_SIZE):
            mini_batch = batch[i:i + BATCH_SIZE]
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(mini_batch)
            else:
                outputs = model(mini_batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            probs_all.extend(probs)
    detections = []
    for box, prob, quality in zip(valid_boxes, probs_all, qualities):
        final_score = float(prob) * float(quality)
        if prob < PATCH_THRESHOLD:
            continue
        if final_score < MIN_FINAL_SCORE:
            continue
        x1, y1, x2, y2 = box
        detections.append({"box": box, "prob": float(prob), "quality": float(quality), "score": float(final_score), "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))})
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    detections = detections[:TOP_K_DETECTIONS]
    return detections


def analyze_frame_blob(frame, model, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)
    proposals = generate_blob_proposals(gray)
    detections = classify_proposals(gray, proposals, model, device)
    scores = [det["score"] for det in detections]
    probs = [det["prob"] for det in detections]
    if len(scores) > 0:
        max_score = float(np.max(scores))
        mean_top3_score = float(np.mean(sorted(scores, reverse=True)[:3]))
        max_prob = float(np.max(probs))
    else:
        max_score = 0.0
        mean_top3_score = 0.0
        max_prob = 0.0
    return proposals, detections, max_prob, max_score, mean_top3_score


class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.boxes = deque(maxlen=TRACK_SCORE_WINDOW)
        self.centers = deque(maxlen=TRACK_SCORE_WINDOW)
        self.probs = deque(maxlen=TRACK_SCORE_WINDOW)
        self.qualities = deque(maxlen=TRACK_SCORE_WINDOW)
        self.scores = deque(maxlen=TRACK_SCORE_WINDOW)
        self.hits = 0
        self.consecutive_hits = 0
        self.missed_frames = 0
        self.update(detection)

    def update(self, detection):
        self.boxes.append(detection["box"])
        self.centers.append(detection["center"])
        self.probs.append(detection["prob"])
        self.qualities.append(detection["quality"])
        self.scores.append(detection["score"])
        self.hits += 1
        self.consecutive_hits += 1
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1
        self.consecutive_hits = 0

    def last_center(self):
        return self.centers[-1]

    def last_box(self):
        return self.boxes[-1]

    def avg_score(self):
        return float(np.mean(self.scores)) if len(self.scores) else 0.0

    def avg_prob(self):
        return float(np.mean(self.probs)) if len(self.probs) else 0.0

    def motion_pixels(self):
        if len(self.centers) < 2:
            return 0.0
        first = self.centers[0]
        last = self.centers[-1]
        return float(np.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2))

    def is_warning_track(self):
        if self.hits < MIN_TRACK_HITS_FOR_WARNING:
            return False
        if self.consecutive_hits < MIN_CONSECUTIVE_HITS_FOR_WARNING:
            return False
        if self.avg_score() < TRACK_CONF_THRESHOLD:
            return False
        if REQUIRE_TRACK_MOTION and self.motion_pixels() < MIN_TRACK_MOTION_PIXELS:
            return False
        return True


def center_distance(p1, p2):
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


class ReflectionTracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 1

    def update(self, detections):
        matched_tracks = set()
        matched_detections = set()
        detection_order = sorted(range(len(detections)), key=lambda i: detections[i]["score"], reverse=True)
        for det_idx in detection_order:
            detection = detections[det_idx]
            best_track_idx = None
            best_distance = float("inf")
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_tracks:
                    continue
                dist = center_distance(detection["center"], track.last_center())
                if dist < best_distance:
                    best_distance = dist
                    best_track_idx = track_idx
            if best_track_idx is not None and best_distance <= MAX_TRACK_DISTANCE:
                self.tracks[best_track_idx].update(detection)
                matched_tracks.add(best_track_idx)
                matched_detections.add(det_idx)
        # mark missed
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track.mark_missed()
        # new tracks
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks.append(Track(detection=detection, track_id=self.next_track_id))
                self.next_track_id += 1
        # remove stale
        self.tracks = [t for t in self.tracks if t.missed_frames <= MAX_MISSED_FRAMES]
        return self.tracks



def run(video_path, model_path, out_csv=None, out_video=None, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(model_path, device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    logs = []
    frames_out = []
    tracker = ReflectionTracker()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        proposals, detections, max_prob, max_score, mean_top3_score = analyze_frame_blob(frame, model, device)
        # tracker expects detections as list of dicts with box, prob, quality, score, center
        tracks = tracker.update(detections)
        # draw detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{det['prob']:.2f}", (x1, max(y1 - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # draw tracks
        warning_tracks = []
        for t in tracks:
            if t.missed_frames > 0:
                continue
            x1, y1, x2, y2 = t.last_box()
            is_warning_track = t.is_warning_track()
            color = (0, 0, 255) if is_warning_track else (0, 180, 255)
            thickness = 3 if is_warning_track else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"T{t.track_id} S:{t.avg_score():.2f} H:{t.hits}", (x1, min(y2 + 18, frame.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            centers = list(t.centers)
            for i in range(1, len(centers)):
                cv2.line(frame, centers[i - 1], centers[i], color, 2)
            if is_warning_track:
                warning_tracks.append(t.track_id)
        warning = len(warning_tracks) >= MIN_WARNING_TRACKS
        if writer:
            writer.write(frame)
        logs.append({"frame": frame_idx, "timestamp_sec": frame_idx / fps if fps > 0 else 0, "num_proposals": len(proposals), "num_detections": len(detections), "active_tracks": len([t for t in tracks if t.missed_frames == 0]), "warning_tracks": len(warning_tracks), "max_prob": max_prob, "max_score": max_score, "mean_top3_score": mean_top3_score, "warning": int(warning)})
        # prepare frame-level JSON-friendly info
        frame_info = {
            "frame": frame_idx,
            "timestamp_sec": frame_idx / fps if fps > 0 else 0,
            "proposals": [list(map(int, p)) for p in proposals],
            "detections": [
                {
                    "box": [int(x) for x in d["box"]],
                    "prob": float(d["prob"]),
                    "quality": float(d["quality"]),
                    "score": float(d["score"]),
                    "center": [int(c) for c in d["center"]],
                }
                for d in detections
            ],
            "tracks": [
                {
                    "track_id": int(t.track_id),
                    "last_box": [int(x) for x in t.last_box()],
                    "avg_score": float(t.avg_score()),
                    "hits": int(t.hits),
                    "missed_frames": int(t.missed_frames),
                    "warning": bool(t.is_warning_track()),
                }
                for t in tracks
            ],
            "warning": bool(warning),
        }
        frames_out.append(frame_info)
        frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    df = pd.DataFrame(logs)
    if out_csv:
        df.to_csv(out_csv, index=False)
    result = {"frames_processed": frame_idx, "detections_total": int(df["num_detections"].sum() if not df.empty else 0), "csv": str(out_csv) if out_csv else None, "video": str(out_video) if out_video else None, "frames": frames_out}
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-video", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    # Prefer an explicit model path, otherwise locate the notebook-trained weights from nearby roots.
    default_model = resolve_model_path(args.model)
    try:
        res = run(args.video, default_model, out_csv=args.out_csv, out_video=args.out_video, device_str=args.device)
        print(json.dumps(res))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}))
        raise
