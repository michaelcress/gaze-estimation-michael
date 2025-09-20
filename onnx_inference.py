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


from collections import deque


def _ort_type_to_np(ort_type: str):
    # onnxruntime uses strings like 'tensor(float)'
    return {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint8)": np.uint8,
        "tensor(int64)": np.int64,
    }.get(ort_type, np.float32)




class EyeContactCalib:
    """
    Learns a simple linear map:
        [pitch_pred, yaw_pred] -> [pitch_tgt, yaw_tgt]
    from a few samples where the user is *looking at the lens* at
    different screen positions.
    """
    def __init__(self):
        # 2x2 scale + 2x1 bias (one map per axis; no cross terms by default)
        self.a_p, self.b_p = 1.0, 0.0  # pitch_out = a_p * pitch_in + b_p
        self.a_y, self.b_y = 1.0, 0.0  # yaw_out   = a_y * yaw_in   + b_y
        self.samples_in  = []  # [(pitch_pred, yaw_pred)]
        self.samples_tgt = []  # [(pitch_tgt , yaw_tgt )]
        self.ready = False

    def add(self, pitch_pred, yaw_pred, pitch_tgt, yaw_tgt):
        self.samples_in.append([pitch_pred, yaw_pred])
        self.samples_tgt.append([pitch_tgt,  yaw_tgt])

    def fit(self, min_samples=10):
        if len(self.samples_in) < min_samples:
            self.ready = False
            return False
        X = np.array(self.samples_in, dtype=np.float32)
        T = np.array(self.samples_tgt, dtype=np.float32)
        # Independent 1D fits (robust enough, avoids overfitting cross-terms)
        # Solve least squares: a*x + b ≈ t
        def solve_1d(x, t):
            A = np.stack([x, np.ones_like(x)], axis=1)  # [N,2]
            sol, *_ = np.linalg.lstsq(A, t, rcond=None)
            a, b = float(sol[0]), float(sol[1])
            # avoid degenerate scales
            if not np.isfinite(a) or abs(a) < 0.2: a = 1.0
            if not np.isfinite(b): b = 0.0
            return a, b
        self.a_p, self.b_p = solve_1d(X[:,0], T[:,0])
        self.a_y, self.b_y = solve_1d(X[:,1], T[:,1])
        self.ready = True
        return True

    def map_pred(self, pitch_pred, yaw_pred):
        pitch_m = self.a_p * pitch_pred + self.b_p
        yaw_m   = self.a_y * yaw_pred   + self.b_y
        return pitch_m, yaw_m

        
        
def target_angles_to_lens(cx_px, cy_px, img_w, img_h, hfov_deg=90.0, vfov_deg=None, mirror_x=False, cx_bias=0.0, cy_bias=0.0, y_up=True):
    c0x = img_w/2.0 + cx_bias
    c0y = img_h/2.0 + cy_bias
    if mirror_x:
        cx_px = (img_w - 1) - cx_px
    dx = cx_px - c0x
    dy = cy_px - c0y
    fx = (img_w/2.0) / np.tan(np.deg2rad(hfov_deg)/2.0)
    if vfov_deg is None:
        vfov = 2*np.rad2deg(np.arctan((img_h/img_w)*np.tan(np.deg2rad(hfov_deg)/2.0)))
    else:
        vfov = vfov_deg
    fy = (img_h/2.0) / np.tan(np.deg2rad(vfov)/2.0)
    yaw_t   = np.arctan2(dx, fx)
    pitch_t = np.arctan2((-dy if y_up else dy), fy)
    return pitch_t, yaw_t




class LookAtYouFilter:
    def __init__(self, ema_alpha=0.3, on_frames=3, off_frames=3, hysteresis_px=8.0):
        self.alpha = ema_alpha
        self.on_need = on_frames
        self.off_need = off_frames
        self.hyst = hysteresis_px
        self.pitch = None
        self.yaw = None
        self.on_count = 0
        self.off_count = 0
        self.state = False  # False=OFF, True=ON

    def smooth_angles(self, pitch, yaw):
        # EMA on radians
        if self.pitch is None:
            self.pitch, self.yaw = float(pitch), float(yaw)
        else:
            a = self.alpha
            self.pitch = (1-a)*self.pitch + a*float(pitch)
            self.yaw   = (1-a)*self.yaw   + a*float(yaw)
        return self.pitch, self.yaw

    def update(self, err_px, tol_px):
        # Turn ON when inside tol
        on_gate  = err_px <= tol_px
        # Turn OFF only when we clearly leave tol by 'hyst' margin
        off_gate = err_px >  tol_px + self.hyst

        if self.state:  # currently ON
            if off_gate:
                self.off_count += 1
                if self.off_count >= self.off_need:
                    self.state = False
                    self.on_count = 0
                    self.off_count = 0
            else:
                self.off_count = 0
        else:           # currently OFF
            if on_gate:
                self.on_count += 1
                if self.on_count >= self.on_need:
                    self.state = True
                    self.on_count = 0
                    self.off_count = 0
            else:
                self.on_count = 0

        return self.state








# ==== TUNABLES (start here) ====
HFOV_DEG = 90.0   # Brio wide = 90, medium = 78, narrow = 65
VFOV_DEG = None   # let code infer from aspect; or set explicitly if you know it
Y_UP     = False #True   # match your draw convention (you used y_up earlier)
MIRROR_X = False  # set True if your displayed frame is horizontally mirrored
CX_BIAS_PX = 0.0  # tweak if lens center != image center (positive = shift right)
CY_BIAS_PX = 0.0  # tweak if lens center != image center (positive = shift down)
BASE_THRESH = 10.0  # deg (close faces)
MAX_THRESH  = 28.0 #22.0  # deg cap for tiny/far faces
REF_FACE    = 120.0 # px reference face size for threshold scaling
# ===============================

def _focal_px_from_fov(img_w, img_h, hfov_deg=60.0, vfov_deg=None):
    fx = (img_w/2.0) / np.tan(np.deg2rad(hfov_deg)/2.0)
    if vfov_deg is None:
        vfov = 2*np.rad2deg(np.arctan((img_h/img_w)*np.tan(np.deg2rad(hfov_deg)/2.0)))
    else:
        vfov = vfov_deg
    fy = (img_h/2.0) / np.tan(np.deg2rad(vfov)/2.0)
    return fx, fy

def gaze_vec_from_angles(pitch, yaw, y_up=True):
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    vx = cp * sy
    vy = (-sp if y_up else sp)
    vz = cp * cy
    v = np.array([vx, vy, vz], dtype=np.float32)
    return v / (np.linalg.norm(v)+1e-9)

# def target_angles_to_lens(cx_px, cy_px, img_w, img_h,
#                           hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG,
#                           mirror_x=MIRROR_X, cx_bias=CX_BIAS_PX, cy_bias=CY_BIAS_PX,
#                           y_up=Y_UP):
#     # principal point (allow bias)
#     c0x = img_w/2.0 + cx_bias
#     c0y = img_h/2.0 + cy_bias

#     # horizontal mirroring (many selfie previews are mirrored)
#     if mirror_x:
#         cx_px = (img_w - 1) - cx_px

#     dx = cx_px - c0x
#     dy = cy_px - c0y

#     fx, fy = _focal_px_from_fov(img_w, img_h, hfov_deg, vfov_deg)
#     yaw_tgt   = np.arctan2(dx, fx)                  # +right
#     pitch_tgt = np.arctan2(( -dy if y_up else dy ), fy)  # +up if y_up
#     return pitch_tgt, yaw_tgt

def angular_error_deg(pitch_a, yaw_a, pitch_b, yaw_b, y_up=Y_UP):
    va = gaze_vec_from_angles(pitch_a, yaw_a, y_up=y_up)
    vb = gaze_vec_from_angles(pitch_b, yaw_b, y_up=y_up)
    ct = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return np.degrees(np.arccos(ct))


def crop_with_padding(frame, x_min, y_min, x_max, y_max, pad_ratio=0.25):
    h, w = frame.shape[:2]
    bw, bh = x_max - x_min, y_max - y_min
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(w, x_max + pad_x)
    y1 = min(h, y_max + pad_y)

    return frame[y0:y1, x0:x1]





# # -- Keep your FOV/intrinsics settings from earlier --
# def _fx_fy(img_w, img_h, hfov_deg=90.0, vfov_deg=None):
#     fx = (img_w/2) / np.tan(np.deg2rad(hfov_deg)/2)
#     if vfov_deg is None:
#         vfov = 2*np.rad2deg(np.arctan((img_h/img_w)*np.tan(np.deg2rad(hfov_deg)/2)))
#     else:
#         vfov = vfov_deg
#     fy = (img_h/2) / np.tan(np.deg2rad(vfov)/2)
#     return fx, fy

def looking_at_camera_px(pitch, yaw, cx, cy, w, h,
                         hfov_deg=90.0, vfov_deg=None,
                         mirror_x=False, y_up=True,
                         base_tol_px=12.0,  # base tolerance in pixels (close faces, near center)
                         size_boost_px=60.0 # extra px tolerance when face is tiny
                         ):
    # principal point
    c0x, c0y = w/2.0, h/2.0
    if mirror_x: cx = (w - 1) - cx
    dx, dy = (cx - c0x), (cy - c0y)              # target pixel offset

    fx, fy = _fx_fy(w, h, hfov_deg, vfov_deg)

    # Predicted pixel offsets from gaze angles (small-angle projection)
    u_pred = fx * np.tan(yaw)                    # right+
    v_pred = fy * np.tan(-pitch if y_up else pitch)  # up+

    # Distance in pixel space between where the gaze points and the lens direction to the face
    err_px = np.hypot(u_pred - dx, v_pred - dy)

    return err_px





def _fx_fy(img_w, img_h, hfov_deg=90.0, vfov_deg=None):
    fx = (img_w/2) / np.tan(np.deg2rad(hfov_deg)/2)
    if vfov_deg is None:
        vfov = 2*np.rad2deg(np.arctan((img_h/img_w)*np.tan(np.deg2rad(hfov_deg)/2)))
    else:
        vfov = vfov_deg
    fy = (img_h/2) / np.tan(np.deg2rad(vfov)/2)
    return fx, fy

def _err_px_one(pitch, yaw, cx, cy, w, h, fx, fy, mirror_x, y_up):
    # principal point
    c0x, c0y = w/2.0, h/2.0
    if mirror_x: cx = (w - 1) - cx
    dx, dy = (cx - c0x), (cy - c0y)
    # project predicted angles to pixel offsets
    u_pred = fx * np.tan(yaw)                      # right + 
    v_pred = fy * np.tan(-pitch if y_up else pitch)  # up + if y_up
    return float(np.hypot(u_pred - dx, v_pred - dy))

def looking_err_px_auto(pitch, yaw, cx, cy, w, h, hfov_deg=90.0, vfov_deg=None):
    fx, fy = _fx_fy(w, h, hfov_deg, vfov_deg)
    best = (1e9, False, False)  # (err, mirror_x, y_up)
    for mirror_x in (False, True):
        for y_up in (False, True):
            e = _err_px_one(pitch, yaw, cx, cy, w, h, fx, fy, mirror_x, y_up)
            if e < best[0]:
                best = (e, mirror_x, y_up)
    return best  # err_px, mirror_x_used, y_up_used



        



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
            # img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
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

    
    cal = EyeContactCalib()
    collect_cal = False  # press 'c' to toggle collection, 'f' to fit



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


    # Loosen thresholds for small faces
    if hasattr(detector, "conf_thresh"): detector.conf_thresh = 0.25  # was 0.5
    if hasattr(detector, "nms_thresh"):  detector.nms_thresh  = 0.45

    # Use a larger detector input (or let it adapt to the frame)
    if hasattr(detector, "dynamic_size"): detector.dynamic_size = True
    if hasattr(detector, "input_size"):   detector.input_size   = (1280, 1280)  # or (1280,1280) if your GPU can handle it





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

    # look_filter = LookAtYouFilter(
    #     ema_alpha=0.25,   # 0.2–0.35 works well
    #     on_frames=3,      # need 3 consecutive good frames to turn on
    #     off_frames=3,     # need 3 bad frames to turn off
    #     hysteresis_px=10  # pixels; prevents flicker
    # )
    look_filter = LookAtYouFilter(
        ema_alpha=0.25,
        on_frames=2,     # was 3
        off_frames=3,
        hysteresis_px=6  # was 10
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, _ = detector.detect(frame)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])

            h, w = frame.shape[:2]
            min_face_px = int(0.025 * min(w, h))  # ~2.5% of min dimension
            min_face_px = np.clip(min_face_px, 48, 120)  # keep sane bounds for 4K

            # Skip small face boxes
            w_box = x_max - x_min
            h_box = y_max - y_min
            if min(w_box, h_box) < min_face_px:
                continue

            # face_crop = frame[y_min:y_max, x_min:x_max]
            face_crop = crop_with_padding(frame, x_min, y_min, x_max, y_max, pad_ratio=0.25)
            if face_crop.size == 0:
                continue

            pitch, yaw = engine.estimate(face_crop)  # radians
            draw_bbox_gaze(frame, bbox, pitch, yaw)



                                

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
