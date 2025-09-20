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



def _fx_fy(img_w, img_h, hfov_deg=90.0, vfov_deg=None):
    fx = (img_w/2.0) / np.tan(np.deg2rad(hfov_deg)/2.0)
    if vfov_deg is None:
        vfov = 2*np.rad2deg(np.arctan((img_h/img_w) * np.tan(np.deg2rad(hfov_deg)/2.0)))
    else:
        vfov = vfov_deg
    fy = (img_h/2.0) / np.tan(np.deg2rad(vfov)/2.0)
    return fx, fy

def gaze_offset_px(pitch, yaw, w, h, hfov_deg=90.0, vfov_deg=None, y_up=False):
    """
    Project the gaze (pitch,yaw in radians) to pixel offsets from the face center.
    Returns (du, dv) in pixels. Length sqrt(du^2+dv^2) is your on-screen vector length.
    """
    fx, fy = _fx_fy(w, h, hfov_deg, vfov_deg)
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



    # Compute a persistent pixel threshold from the chosen %. Recompute if resolution changes.
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Could not grab first frame to size threshold.")
    h0, w0 = frame0.shape[:2]
    L_thresh_px = float(np.clip(L_THRESH_INIT * min(w0, h0), L_THRESH_MIN_PX, L_THRESH_MAX_PX))

    # If you want to reuse that first frame, show it, then continue as usual:
    cv2.imshow("Gaze Estimation", frame0)



    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, _ = detector.detect(frame)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue

            pitch, yaw = engine.estimate(face_crop)  # radians
            draw_bbox_gaze(frame, bbox, pitch, yaw)

            # # Build 3D gaze vector (unit length), compare to camera forward (0,0,1)
            # gvec = gaze_vector_from_angles(pitch, yaw, convention="y_up")
            # looking, angle_deg = is_looking_at_camera(gvec, thresh_deg=10.0)  # tweak threshold 8–12°

            # if looking:
            #     # Optional: also check that the face center is near image center (helps avoid false positives)
            #     h, w = frame.shape[:2]
            #     cx = (x_min + x_max) // 2
            #     cy = (y_min + y_max) // 2
            #     # distance from center normalized by min dimension
            #     d = np.hypot(cx - w//2, cy - h//2) / max(1, min(w, h))
            #     if d < 0.2:  # tweak 0.15–0.25 depending on your FOV
            #         cv2.putText(frame, "Looking at you",
            #                     (x_min, max(0, y_min - 10)),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)


            # # Clamp & (optionally) smooth your angles
            # pitch = float(np.clip(pitch, np.deg2rad(-42), np.deg2rad(42)))
            # yaw   = float(np.clip(yaw,   np.deg2rad(-42), np.deg2rad(42)))
            # pitch_s, yaw_s = look_filter.smooth_angles(pitch, yaw)  # reuse your EMA

                        
            # Clamp angles to model range before projecting (avoids tan blowups)
            pitch_c = float(np.clip(pitch, np.deg2rad(-42), np.deg2rad(42)))
            yaw_c   = float(np.clip(yaw,   np.deg2rad(-42), np.deg2rad(42)))

            h, w = frame.shape[:2]
            du, dv = gaze_offset_px(pitch_c, yaw_c, w, h, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG, y_up=Y_UP)
            L = (du*du + dv*dv) ** 0.5

            is_on = (L <= L_thresh_px)  # simple, flicker-free rule

            cv2.putText(frame, f"L:{L:.1f} thr:{L_thresh_px:.1f} {'ON' if is_on else 'off'}",
                        (x_min, max(0, y_min - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)

            if is_on:
                cv2.putText(frame, "Looking at you",
                            (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

                                

        if writer:
            writer.write(frame)

        cv2.imshow("Gaze Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(']'):
            L_thresh_px *= 1.1    # looser
        elif key == ord('['):
            L_thresh_px /= 1.1    # stricter
            L_thresh_px = float(np.clip(L_thresh_px, L_THRESH_MIN_PX, L_THRESH_MAX_PX))

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
