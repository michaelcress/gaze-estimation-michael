# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import uniface
import argparse

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

import onnxruntime as ort

from typing import Tuple

from utils.helpers import draw_bbox_gaze


from pathlib import Path



import itertools
from dataclasses import dataclass

import time




def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax1-ax0) * max(0, ay1-ay0)
    area_b = max(0, bx1-bx0) * max(0, by1-by0)
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)

@dataclass
class Track:
    track_id: int
    bbox: tuple  # (x0,y0,x1,y1)
    ema_pitch: float | None = None
    ema_yaw: float | None = None
    last_seen: int = 0

    # dwell-time bookkeeping
    looking_on: bool = False
    on_started_ts: float | None = None   # when the current ON segment began
    on_total_ms: float = 0.0             # cumulative ON milliseconds
    last_update_ts: float | None = None  # last time we updated this track
    current_dwell_duration: float = 0.0



class FaceTracker:
    def __init__(self, iou_thresh=0.3, max_age=15, alpha=0.2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.alpha = alpha
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_idx = 0
        self.max_dwell_threshold_S = 3000

    def update_dwell(self, track_id: int, is_on: bool, now_ts: float | None = None):
        now = time.monotonic() if now_ts is None else now_ts
        tr = self.tracks.get(track_id)
        if tr is None:
            return

        # If we’ve been ON, accumulate time since last update
        if tr.last_update_ts is not None and tr.looking_on:
            tr.on_total_ms += max(0.0, (now - tr.last_update_ts) * 1000.0)
            tr.current_dwell_duration += max(0.0, (now - tr.last_update_ts) * 1000.0)

        if tr.current_dwell_duration > self.max_dwell_threshold_S:
            print("You've been looking for too long")

        # Edge transitions
        if not tr.looking_on and is_on:
            tr.on_started_ts = now
            tr.looking_on = True
        elif tr.looking_on and not is_on:
            # close the segment (already accumulated via last_update_ts)
            tr.on_started_ts = None
            tr.looking_on = False
            tr.current_dwell_duration = 0.0

        tr.last_update_ts = now

    def _smooth(self, tr: Track, pitch, yaw):
        if tr.ema_pitch is None:
            tr.ema_pitch, tr.ema_yaw = float(pitch), float(yaw)
        else:
            a = self.alpha
            tr.ema_pitch = (1-a)*tr.ema_pitch + a*float(pitch)
            tr.ema_yaw   = (1-a)*tr.ema_yaw   + a*float(yaw)
        return tr.ema_pitch, tr.ema_yaw

    def update(self, detections):
        """detections: list of (x0,y0,x1,y1) ints"""
        self.frame_idx += 1

        # mark all as unmatched initially
        unmatched_det = list(range(len(detections)))
        unmatched_trk = list(self.tracks.keys())
        matches = []

        # greedy IoU matching
        if self.tracks and detections:
            pairs = []
            for ti in self.tracks:
                tb = self.tracks[ti].bbox
                for di, db in enumerate(detections):
                    pairs.append((1.0 - iou_xyxy(tb, db), ti, di))  # sort by (1 - IoU)
            pairs.sort()
            used_t, used_d = set(), set()
            for cost, ti, di in pairs:
                if ti in used_t or di in used_d: 
                    continue
                iou = 1.0 - cost
                if iou >= self.iou_thresh:
                    matches.append((ti, di))
                    used_t.add(ti); used_d.add(di)
            unmatched_trk = [ti for ti in self.tracks if ti not in used_t]
            unmatched_det = [di for di in range(len(detections)) if di not in used_d]

        # update matched tracks
        for ti, di in matches:
            self.tracks[ti].bbox = detections[di]
            self.tracks[ti].last_seen = self.frame_idx

        # create new tracks for unmatched detections
        for di in unmatched_det:
            ti = self._next_id; self._next_id += 1
            self.tracks[ti] = Track(track_id=ti, bbox=detections[di], last_seen=self.frame_idx)

        # GC stale tracks
        # to_del = [ti for ti, tr in self.tracks.items()
        #           if (self.frame_idx - tr.last_seen) > self.max_age]
        # for ti in to_del: del self.tracks[ti]
        to_del = []
        for ti, tr in self.tracks.items():
            if (self.frame_idx - tr.last_seen) > self.max_age:
                # finalize any open ON segment
                if tr.looking_on and tr.last_update_ts is not None:
                    tr.on_total_ms += max(0.0, (time.monotonic() - tr.last_update_ts) * 1000.0)
                to_del.append(ti)
        for ti in to_del:
            del self.tracks[ti]

        # return list of (track_id, bbox)
        return [(ti, self.tracks[ti].bbox) for ti in self.tracks]

    def smooth_angles_for(self, track_id, pitch, yaw):
        return self._smooth(self.tracks[track_id], pitch, yaw)




def _ort_type_to_np(ort_type: str):
    # onnxruntime uses strings like 'tensor(float)'
    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint8)": np.uint8,
        "tensor(int64)": np.int64,
    }.get(ort_type, np.float32)



# --- Simple gaze-length mode tunables ---
HFOV_DEG   = 90.0     # Brio wide=90, medium=78, narrow=65
VFOV_DEG   = None     # infer from aspect
Y_UP       = False    # OpenCV draw is y-down; keep False unless you draw y-up

# Initial threshold: ~2% of the short side (22px @1080p, 43px @4K)
L_THRESH_INIT = 0.02

# Optional: hard caps for threshold in pixels
L_THRESH_MIN_PX = 18.0
L_THRESH_MAX_PX = 60.0






# --- False-positive filters (add near other tunables) ---
DET_CONF_THRESH   = 0.80     # raise if you still see wall hits (0.85–0.90)
MIN_FACE_PX       = 64       # min box side (px); raise if camera is close
MAX_REL_FACE      = 0.45     # max box fraction of min(frame_w, frame_h)

# If the wall artifact is always in the same area, exclude it:
# Set to None to disable, or fill in (x, y, w, h)
EXCLUDE_RECT      = None     # e.g., (400, 120, 200, 180)

# Texture filter: flat / low-variance ROIs are likely not real faces
TEXTURE_VAR_MIN   = 70.0     # bump to ~90–120 if walls still slip through

EXCLUDE_RECT_COLOR = (255, 0, 255)   # magenta in BGR (blue=255, green=0, red=255)
EXCLUDE_RECT_THICK = 2
EXCLUDE_FILL_ALPHA = 0.18            # optional translucent fill






def in_excluded_region(box, excl_rect):
    if excl_rect is None:
        return False
    x0, y0, x1, y1 = box
    ex, ey, ew, eh = excl_rect
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return (ex <= cx <= ex + ew) and (ey <= cy <= ey + eh)

def size_ok_xyxy(box, frame_shape):
    x0, y0, x1, y1 = box
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    if min(w, h) < MIN_FACE_PX:
        return False
    max_abs = int(MAX_REL_FACE * min(frame_shape[1], frame_shape[0]))
    return (w <= max_abs) and (h <= max_abs)

def texture_ok_gray(gray, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(gray.shape[1], x1); y1 = min(gray.shape[0], y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    lap_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    return lap_var >= TEXTURE_VAR_MIN







def _fx_fy(img_w, img_h, hfov_deg=90.0, vfov_deg=None):
    fx = (img_w/2.0) / np.tan(np.deg2rad(hfov_deg)/2.0)
    if vfov_deg is None:
        vfov = 2*np.rad2deg(np.arctan((img_h/img_w) * np.tan(np.deg2rad(hfov_deg)/2.0)))
    else:
        vfov = vfov_deg
    fy = (img_h/2.0) / np.tan(np.deg2rad(vfov)/2.0)
    return fx, fy

# def gaze_offset_px(pitch, yaw, w, h, hfov_deg=90.0, vfov_deg=None, y_up=False):
#     """
#     Project the gaze (pitch,yaw in radians) to pixel offsets from the face center.
#     Returns (du, dv) in pixels. Length sqrt(du^2+dv^2) is your on-screen vector length.
#     """
#     u = fx * np.tan(yaw)                         # right +
#     v = fy * np.tan(-pitch if y_up else pitch)   # up + if y_up, else y-down
#     return float(u), float(v)
def gaze_offset_px(pitch, yaw, fx, fy, y_up=False):
    u = fx * np.tan(yaw)                         # right +
    v = fy * np.tan(-pitch if y_up else pitch)   # up + if y_up, else y-down
    return float(u), float(v)




def gaze_vector_from_angles(pitch_rad: float, yaw_rad: float, convention="y_up"):
    """
    Convert (pitch,yaw) -> 3D unit vector.
    - pitch: up(+)/down(-) in radians
    - yaw:   right(+)/left(-) in radians
    convention:
      "y_up":  x right, y up, z forward
      "y_down": x right, y down, z forward
    """
    cp = np.cos(pitch_rad); sp = np.sin(pitch_rad)
    cy = np.cos(yaw_rad);   sy = np.sin(yaw_rad)

    # x right, z forward, choose y sign by convention
    vx = cp * sy
    vy = (-sp if convention == "y_up" else sp)
    vz = cp * cy
    v = np.array([vx, vy, vz], dtype=np.float32)
    # Normalize just in case
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def is_looking_at_camera(gaze_vec: np.ndarray, fwd=np.array([0,0,1],dtype=np.float32), thresh_deg=10.0):
    """
    True if angle between gaze_vec and camera forward is within thresh_deg.
    """
    gaze = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-9)
    fwd  = fwd / (np.linalg.norm(fwd) + 1e-9)
    cos_t = float(np.clip(np.dot(gaze, fwd), -1.0, 1.0))
    theta = np.degrees(np.arccos(cos_t))
    return theta <= thresh_deg, theta


def _softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

class GazeEstimationONNX:
    """
    Gaze estimation using ONNXRuntime (logits -> degrees via expectation over bins).
    """

    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        self.session = session
        if self.session is None:
            assert model_path is not None, "Model path is required for the first time initialization."
            model_path = str(Path(model_path).expanduser().resolve())

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            trt_options = {
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
            }
            providers = [
                ("TensorrtExecutionProvider", trt_options),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
            print("Using:", self.session.get_providers())

        # from your dataset config for mpiigaze
        self.bins = 28
        self.binwidth = 3.0           # degrees per bin
        self.angle_half_range = 42.0  # i.e., angles in [-42, +42]
        self.angle_min = -self.angle_half_range
        self.angle_max =  self.angle_half_range

        # Read input metadata from the model (handles dynamic N/H/W)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        ishape = inp.shape  # e.g., [None, 3, 448, 448] or [1,3,448,448]
        # Fallback if model is dynamic; otherwise use graph values:
        H = ishape[2] if isinstance(ishape[2], int) else 448
        W = ishape[3] if isinstance(ishape[3], int) else 448
        self.input_size = (W, H)

        # (Optional) keep your normalization
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std  = [0.229, 0.224, 0.225]

        # Verify there are 2 outputs (pitch/yaw)
        outs = self.session.get_outputs()
        self.output_names = [o.name for o in outs]

        names_l = [n.lower() for n in self.output_names]
        if "pitch" in names_l[0] and "yaw" in names_l[1]:
            self.pitch_idx, self.yaw_idx = 0, 1
        elif "yaw" in names_l[0] and "pitch" in names_l[1]:
            self.pitch_idx, self.yaw_idx = 1, 0
        else:
            print("Unable to determine pitch and yaw names indices. Assuming pitch=0 and yaw = 1\n")
            self.pitch_idx, self.yaw_idx = 0, 1  # fallback to your current order


        assert len(self.output_names) == 2, f"Expected 2 outputs, got {len(self.output_names)}: {self.output_names}"

    def decode(self, pitch_logits, yaw_logits):
        pitch_probs = _softmax(pitch_logits, axis=1)
        yaw_probs   = _softmax(yaw_logits,   axis=1)

        B = pitch_probs.shape[1]
        assert B == self.bins, f"Expected {self.bins} bins, got {B}"

        centers = self.angle_min + (np.arange(B, dtype=np.float32) + 0.5) * self.binwidth  # degrees
        pitch_deg = (pitch_probs @ centers).astype(np.float32)
        yaw_deg   = (yaw_probs   @ centers).astype(np.float32)

        # 🔑 Return radians (match your original code & likely what draw_bbox_gaze expects)
        return np.deg2rad(pitch_deg[0]).item(), np.deg2rad(yaw_deg[0]).item()

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Convert HxWxC uint8 face crop -> model input tensor with shape [1, C, H, W] or [1, H, W, C]."""
        inp = self.session.get_inputs()[0]
        ishape = inp.shape  # e.g. [1,3,448,448] (NCHW) or [1,448,448,3] (NHWC) or dynamic dims
        ort_dtype = _ort_type_to_np(inp.type)

        # Determine layout
        if len(ishape) == 4 and (ishape[1] in (1, 3)):
            layout = "NCHW"
            C = ishape[1] if isinstance(ishape[1], int) else 3
            H = ishape[2] if isinstance(ishape[2], int) else self.input_size[1]
            W = ishape[3] if isinstance(ishape[3], int) else self.input_size[0]
        else:
            layout = "NHWC"
            C = ishape[3] if (len(ishape) == 4 and isinstance(ishape[3], int)) else 3
            H = ishape[1] if (len(ishape) == 4 and isinstance(ishape[1], int)) else self.input_size[1]
            W = ishape[2] if (len(ishape) == 4 and isinstance(ishape[2], int)) else self.input_size[0]

        # Ensure 3-channel RGB uint8 image
        img = face_crop
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # If image likely from OpenCV (BGR), convert to RGB
        if cv2 is not None and isinstance(img, np.ndarray):
            # Heuristic: if caller passed in OpenCV frames, they’re BGR; flip to RGB
            img = img[:, :, ::-1]

        # Resize
        if cv2 is not None:
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = np.array(Image.fromarray(img).resize((W, H), Image.BILINEAR))

        # To float32 in [0,1]
        x = img_resized.astype(np.float32) / 255.0

        # Normalize (ImageNet)
        mean = np.array(self.input_mean, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(self.input_std,  dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std

        # Layout + batch
        if layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))           # HWC -> CHW
            x = np.expand_dims(x, axis=0)            # -> NCHW
        else:
            x = np.expand_dims(x, axis=0)            # -> NHWC

        # Cast to expected dtype (usually float32)
        x = x.astype(ort_dtype, copy=False)
        return np.ascontiguousarray(x)

    def estimate(self, face_crop: np.ndarray):
        x = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: x})

        # If order is unknown, pick by name (optional)
        # names = self.output_names
        # pitch_idx = 0 if 'pitch' in names[0].lower() else 1
        # yaw_idx   = 1 - pitch_idx
        # pitch_logits, yaw_logits = outputs[pitch_idx], outputs[yaw_idx]

        pitch_logits, yaw_logits = outputs[self.pitch_idx], outputs[self.yaw_idx]

        # print("logits shapes (pitch, yaw):", pitch_logits.shape, yaw_logits.shape)  # expect (1, 28) each
        pitch_deg, yaw_deg = self.decode(pitch_logits, yaw_logits)
        # print("pitch/yaw (deg):", np.rad2deg(pitch_deg), np.rad2deg(yaw_deg))  # should be within ~[-42, 42]

        return pitch_deg, yaw_deg


REF_FACE  = 120.0   # px
THR_CLOSE = 130.0   # px when face≈REF_FACE
THR_FAR   = 35.0    # px cap for tiny/far faces

def gaze_thresh_px(face_px: float) -> float:
    scale = face_px / REF_FACE
    return float(np.clip(THR_CLOSE * scale, THR_FAR, THR_CLOSE))



def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation ONNX Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path or camera index (e.g., 0 for webcam)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video (optional)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle numeric webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {args.source}")

    # Initialize Gaze Estimation model
    engine = GazeEstimationONNX(model_path=args.model)


    def _initialize_model_gpu(self, model_path: str):
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("RetinaFace providers:", self.session.get_providers())

    # Patch the class method then construct
    uniface.RetinaFace._initialize_model = _initialize_model_gpu
    detector = uniface.RetinaFace()

    print("Gaze providers  :", engine.session.get_providers())
    print("Retina providers:", detector.session.get_providers())


    # Optional output writer
    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))


    fx, fy = None, None  # camera intrinsics in pixels (focal lengths)
    w0, h0 = None, None  # last-seen frame size

    # Compute a persistent pixel threshold from the chosen %. Recompute if resolution changes.
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Could not grab first frame to size threshold.")

    h0, w0 = frame0.shape[:2]
    fx, fy = _fx_fy(w0, h0, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG)

    # L_thresh_px = float(np.clip(L_THRESH_INIT * min(w0, h0), L_THRESH_MIN_PX, L_THRESH_MAX_PX))

    # If you want to reuse that first frame, show it, then continue as usual:
    cv2.imshow("Gaze Estimation", frame0)


    tracker = FaceTracker(iou_thresh=0.35, max_age=20, alpha=0.22)  # tweak alpha 0.15–0.3
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        h, w = frame.shape[:2]
        if (w != w0) or (h != h0):
            # resolution changed (e.g., camera renegotiated)
            fx, fy = _fx_fy(w, h, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG)
            w0, h0 = w, h
            

        bboxes, _ = detector.detect(frame)

        # Prepare gray once for texture check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = []
        for b in bboxes:
            # Be robust to output shapes: [x0,y0,x1,y1,score?]
            if len(b) >= 5:
                x0, y0, x1, y1, score = b[:5]
            else:
                x0, y0, x1, y1 = b[:4]
                score = 1.0  # if API doesn’t return score, treat as 1.0 (still filtered by texture/size/mask)

            # 1) confidence gate
            if score < DET_CONF_THRESH:
                continue

            box = (int(x0), int(y0), int(x1), int(y1))

            # 2) region exclude (fixed wall spot)
            if in_excluded_region(box, EXCLUDE_RECT):
                continue

            # 3) size sanity
            if not size_ok_xyxy(box, frame.shape):
                continue

            # 4) texture / variance
            if not texture_ok_gray(gray, box):
                continue

            dets.append(box)

        # Optionally draw excluded region for debug
        if EXCLUDE_RECT is not None:
            ex, ey, ew, eh = EXCLUDE_RECT
            # outline
            cv2.rectangle(
                frame, (ex, ey), (ex + ew, ey + eh),
                EXCLUDE_RECT_COLOR, EXCLUDE_RECT_THICK, lineType=cv2.LINE_AA
            )

            # optional translucent fill so it really stands out
            if EXCLUDE_FILL_ALPHA and EXCLUDE_FILL_ALPHA > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), EXCLUDE_RECT_COLOR, thickness=-1)
                frame[:] = cv2.addWeighted(overlay, EXCLUDE_FILL_ALPHA, frame, 1 - EXCLUDE_FILL_ALPHA, 0)

        tracked = tracker.update(dets)



        for track_id, (x_min, y_min, x_max, y_max) in tracked:
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue

            # Skip tiny faces (helps stability)
            w_box = x_max - x_min; h_box = y_max - y_min
            if min(w_box, h_box) < 64:
                continue

            # Estimate raw angles
            pitch, yaw = engine.estimate(face_crop)  # radians

            # Clamp, then EMA per-track
            pitch_c = float(np.clip(pitch, np.deg2rad(-42), np.deg2rad(42)))
            yaw_c   = float(np.clip(yaw,   np.deg2rad(-42), np.deg2rad(42)))
            pitch_s, yaw_s = tracker.smooth_angles_for(track_id, pitch_c, yaw_c)

            # Draw using smoothed values (optional: pass smoothed to your drawer)
            draw_bbox_gaze(frame, (x_min, y_min, x_max, y_max), pitch_s, yaw_s)

            # Project to pixels with cached fx, fy
            du, dv = gaze_offset_px(pitch_s, yaw_s, fx, fy, y_up=Y_UP)
            L = (du*du + dv*dv) ** 0.5

            # Threshold scales with face size (your function)
            face_sz = float(min(w_box, h_box))
            L_thresh = gaze_thresh_px(face_sz)
            is_on = (L <= L_thresh)

            tracker.update_dwell(track_id, is_on)


            # Debug overlay with track id
            secs = tracker.tracks[track_id].on_total_ms / 1000.0
            cv2.putText(frame, f"ID:{track_id} look:{'ON' if is_on else 'off'} {secs:0.1f}s",
                        (x_min, max(0, y_min-46)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2, cv2.LINE_AA)

            # cv2.putText(frame, f"ID:{track_id} L:{L:.1f} thr:{L_thresh:.1f}",
            #             (x_min, max(0, y_min-28)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)
            if is_on:
                cv2.putText(frame, "Looking at you",
                            (x_min, max(0, y_min-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        # for bbox in bboxes:
            # x_min, y_min, x_max, y_max = map(int, bbox[:4])
            # face_crop = frame[y_min:y_max, x_min:x_max]
            # if face_crop.size == 0:
            #     continue

                
            # w_box = x_max - x_min
            # h_box = y_max - y_min

            # # Skip small faces
            # if min(w_box, h_box) < 64: # 48   # ~48 for 1080p; consider 64–96 for 4K
            #     continue
                


            # pitch, yaw = engine.estimate(face_crop)  # radians
            # draw_bbox_gaze(frame, bbox, pitch, yaw)

                        
            # # Clamp angles to model range before projecting (avoids tan blowups)
            # pitch_c = float(np.clip(pitch, np.deg2rad(-42), np.deg2rad(42)))
            # yaw_c   = float(np.clip(yaw,   np.deg2rad(-42), np.deg2rad(42)))

            # # Project gaze to pixel offsets
            # du, dv = gaze_offset_px(pitch_c, yaw_c, fx, fy, y_up=Y_UP)
            # L = (du*du + dv*dv) ** 0.5


            # # Current face box size
            # face_sz = float(min(w_box, h_box))

            # # Scale factor: bigger faces → higher threshold
            # scale = face_sz / REF_FACE

            # # Interpolate threshold between far and close
            # L_thresh = gaze_thresh_px(face_sz)


            # is_on = (L <= L_thresh)
           

            # cv2.putText(frame, f"L:{L:.1f} thr:{L_thresh:.1f} {'ON' if is_on else 'off'}",
            #             (x_min, max(0, y_min - 28)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)

            # if is_on:
            #     cv2.putText(frame, "Looking at you",
            #                 (x_min, max(0, y_min - 10)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)


        if writer:
            writer.write(frame)

        cv2.imshow("Gaze Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
