import face_recognition
import cv2
import os
import math
import time
import numpy as np
import threading
import requests
import logging
from datetime import datetime
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("CFRS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CFRS")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_ALLOWED_BACKEND_SCHEMES = ("http", "https")
_MAX_TRACK_ID = 2**31 - 1  # Prevent unbounded growth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_url(url: str, allowed_schemes=_ALLOWED_BACKEND_SCHEMES) -> str:
    """Validate a URL to prevent SSRF and other injection attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Allowed: {allowed_schemes}"
        )
    if not parsed.hostname:
        raise ValueError(f"URL '{url}' has no valid hostname.")
    # Block obvious internal metadata endpoints (cloud metadata services)
    blocked_hosts = ("169.254.169.254", "metadata.google.internal")
    if parsed.hostname in blocked_hosts:
        raise ValueError(f"URL hostname '{parsed.hostname}' is blocked for security.")
    return url


def _sanitize_path(path: str, must_be_under: str | None = None) -> str:
    """Resolve a filesystem path and optionally ensure it stays under a root."""
    resolved = os.path.realpath(path)
    if must_be_under:
        root = os.path.realpath(must_be_under)
        if not resolved.startswith(root + os.sep) and resolved != root:
            raise ValueError(
                f"Path '{path}' resolves outside allowed root '{must_be_under}'."
            )
    return resolved


def _write_dashboard_frame(frame_output_path: str, jpeg_bytes: bytes) -> None:
    """Atomically write a JPEG frame for the dashboard with retry logic."""
    tmp_path = f"{frame_output_path}.tmp"

    for attempt in range(4):
        try:
            with open(tmp_path, "wb") as fp:
                fp.write(jpeg_bytes)
            os.replace(tmp_path, frame_output_path)
            return
        except (PermissionError, OSError):
            time.sleep(0.02 * (attempt + 1))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    # Fallback: direct write when atomic replace is blocked by transient lock.
    try:
        with open(frame_output_path, "wb") as fp:
            fp.write(jpeg_bytes)
    except OSError as exc:
        logger.warning("Dashboard frame fallback write failed: %s", exc)


# ---------------------------------------------------------------------------
# Core Service
# ---------------------------------------------------------------------------
class ClassroomFacialRecognitionService:

    def __init__(self, known_faces_path: str = "known_faces"):
        self.path = _sanitize_path(known_faces_path)
        self.known_db: dict[str, list[np.ndarray]] = {}

        # Matching weights & thresholds
        self.WEIGHT_BEST = 0.7
        self.WEIGHT_SECOND = 0.3
        self.CONFIDENCE_K = 12
        self.MIN_CONFIDENCE = 60.0

        # Blur detection
        self.BLUR_MIN_THRESH = 40.0
        self.BLUR_MAX_THRESH = 60.0
        self.BLUR_FACE_RATIO = 800.0

        # Behavior classification
        self.EAR_THRESH = float(os.getenv("CFRS_EAR_THRESH", "0.20"))
        self.POSE_INATTENTIVE_PENALTY = float(
            os.getenv("CFRS_POSE_INATTENTIVE_PENALTY", "0.06")
        )

        # Body detector
        self.body_detector = cv2.HOGDescriptor()
        self.body_detector.setSVMDetector(
            cv2.HOGDescriptor_getDefaultPeopleDetector()
        )

        self._load_and_encode_database()

    # ------------------------------------------------------------------
    # Database loading
    # ------------------------------------------------------------------
    def _load_and_encode_database(self) -> None:
        """Index known faces with num_jitters=5 and robust error handling."""
        logger.info("Booting Identity Service...")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.warning(
                "Directory created. Please add images to '%s'.", self.path
            )
            return

        for filename in os.listdir(self.path):
            if not filename.lower().endswith(_ALLOWED_IMAGE_EXTENSIONS):
                continue

            name = os.path.splitext(filename)[0].split("_")[0].upper().strip()
            if not name:
                logger.warning("Empty name derived from '%s'. Skipping.", filename)
                continue

            img_path = os.path.join(self.path, filename)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning("Cannot read '%s'. Skipping.", filename)
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(
                    rgb_img, num_jitters=5
                )

                if encodings:
                    self.known_db.setdefault(name, []).append(encodings[0])
                    logger.info(
                        "  -> Learned face from %s (High-Quality Jitter)", filename
                    )
                else:
                    logger.warning("  -> No face detected in %s", filename)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", filename, exc)

        logger.info("Ready! Loaded %d identities.", len(self.known_db))

    # ------------------------------------------------------------------
    # Image quality
    # ------------------------------------------------------------------
    def _check_blur_dynamic(
        self, rgb_image: np.ndarray, face_loc: tuple
    ) -> bool:
        """Return True if the face crop is sharp enough for recognition."""
        top, right, bottom, left = face_loc
        face_width = right - left
        face_crop = rgb_image[top:bottom, left:right]
        if face_crop.size == 0:
            return False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        dynamic_blur_thresh = min(
            self.BLUR_MAX_THRESH,
            max(self.BLUR_MIN_THRESH, self.BLUR_FACE_RATIO / max(face_width, 1.0)),
        )
        return variance > dynamic_blur_thresh

    # ------------------------------------------------------------------
    # Pose & attention
    # ------------------------------------------------------------------
    def _get_pose_penalty(self, face_landmarks: dict | None) -> float:
        """Compute pose penalty from pre-computed landmarks (no per-face re-calc)."""
        if not face_landmarks:
            return 0.0

        lm = face_landmarks
        if "left_eye" in lm and "right_eye" in lm and "nose_bridge" in lm:
            left_eye_center = np.mean(lm["left_eye"], axis=0)
            right_eye_center = np.mean(lm["right_eye"], axis=0)
            nose_center = np.mean(lm["nose_bridge"], axis=0)

            dist_left = np.linalg.norm(left_eye_center - nose_center)
            dist_right = np.linalg.norm(right_eye_center - nose_center)

            ratio = min(dist_left, dist_right) / max(
                dist_left, dist_right, 1e-6
            )
            if ratio < 0.4:
                return 0.08
            if ratio < 0.6:
                return 0.04
        return 0.0

    def get_dynamic_threshold_and_margin(
        self, face_width: float, pose_penalty: float
    ) -> tuple[float, float]:
        face_width = max(20, min(100, face_width))
        t = (face_width - 20) / (100 - 20)
        base_thresh = 0.60 - (0.10 * t)
        margin = 0.08 - (0.06 * t)
        pose_penalty = max(0.0, min(0.1, pose_penalty))
        final_thresh = base_thresh - pose_penalty
        final_thresh = max(0.45, min(0.65, final_thresh))
        return round(final_thresh, 3), round(margin, 3)

    # ------------------------------------------------------------------
    # Distance & confidence
    # ------------------------------------------------------------------
    def _get_weighted_distance(
        self, known_encodings: list[np.ndarray], target_encoding: np.ndarray
    ) -> float:
        if not known_encodings:
            return float("inf")  # FIX: guard against empty list

        dists = face_recognition.face_distance(known_encodings, target_encoding)
        if len(dists) == 0:
            return float("inf")
        if len(dists) == 1:
            return float(dists[0])

        sorted_dists = np.sort(dists)
        weighted_dist = (
            sorted_dists[0] * self.WEIGHT_BEST
            + sorted_dists[1] * self.WEIGHT_SECOND
        )
        return float(weighted_dist)

    def _calculate_confidence(self, distance: float, threshold: float) -> float:
        """Sigmoid confidence mapped to [0.0, 99.9]."""
        # FIX: original had max(1.0, min(99.9, ...)) which clamped low values
        # to 1.0 instead of allowing near-zero confidence. Fixed to max(0.0, ...).
        try:
            confidence = 1 / (1 + math.exp(self.CONFIDENCE_K * (distance - threshold)))
        except OverflowError:
            confidence = 0.0
        return round(max(0.0, min(99.9, confidence * 100)), 2)

    # ------------------------------------------------------------------
    # Eye aspect ratio & attention
    # ------------------------------------------------------------------
    def _eye_aspect_ratio(self, eye_points: list) -> float:
        if not eye_points or len(eye_points) < 6:
            return 0.0
        p = np.array(eye_points, dtype=np.float32)
        a = np.linalg.norm(p[1] - p[5])
        b = np.linalg.norm(p[2] - p[4])
        c = np.linalg.norm(p[0] - p[3])
        if c <= 1e-6:
            return 0.0
        return float((a + b) / (2.0 * c))

    def _classify_attention_state(
        self, face_landmarks: dict | None, pose_penalty: float = 0.0
    ) -> str:
        if not face_landmarks:
            return "ไม่ทราบสถานะ"

        if pose_penalty >= self.POSE_INATTENTIVE_PENALTY:
            return "ไม่ตั้งใจเรียน"

        left_eye = face_landmarks.get("left_eye")
        right_eye = face_landmarks.get("right_eye")
        if not left_eye or not right_eye:
            return "ไม่ทราบสถานะ"

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        return "หลับ/เหม่อ" if avg_ear < self.EAR_THRESH else "ตั้งใจเรียน"

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, resize_scale: float = 1
    ) -> list[dict]:
        small_frame = cv2.resize(
            frame, (0, 0), fx=resize_scale, fy=resize_scale
        )
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(
            rgb_small, face_locations, num_jitters=1
        )
        all_face_landmarks = face_recognition.face_landmarks(
            rgb_small, face_locations
        )

        detections: list[dict] = []

        for idx, (encoding, face_loc) in enumerate(
            zip(face_encodings, face_locations)
        ):
            top, right, bottom, left = face_loc
            face_width = right - left
            scale_back = 1.0 / resize_scale
            y1, x2, y2, x1 = [int(v * scale_back) for v in face_loc]
            bbox = {"Top": y1, "Right": x2, "Bottom": y2, "Left": x1}
            current_landmarks = (
                all_face_landmarks[idx]
                if idx < len(all_face_landmarks)
                else None
            )
            pose_penalty = self._get_pose_penalty(current_landmarks)
            behavior_state = self._classify_attention_state(
                current_landmarks, pose_penalty=pose_penalty
            )

            if not self._check_blur_dynamic(rgb_small, face_loc):
                detections.append(
                    {
                        "Name": "Moving/Blur",
                        "Confidence": 0.0,
                        "BoundingBox": bbox,
                        "State": "ไม่ทราบสถานะ",
                    }
                )
                continue

            threshold, required_margin = self.get_dynamic_threshold_and_margin(
                face_width, pose_penalty
            )

            person_distances: dict[str, float] = {}
            for name, known_encodings in self.known_db.items():
                person_distances[name] = self._get_weighted_distance(
                    known_encodings, encoding
                )

            sorted_persons = sorted(person_distances.items(), key=lambda x: x[1])

            name = "Unknown"
            conf = 0.0

            if sorted_persons:
                best_match_name, best_dist = sorted_persons[0]

                actual_margin = 1.0
                if len(sorted_persons) > 1:
                    actual_margin = sorted_persons[1][1] - best_dist

                conf = self._calculate_confidence(best_dist, threshold)

                if (
                    best_dist <= threshold
                    and actual_margin > required_margin
                    and conf >= self.MIN_CONFIDENCE
                ):
                    name = best_match_name
                else:
                    name = "Unknown"

            detections.append(
                {
                    "Name": name,
                    "Confidence": conf,
                    "BoundingBox": bbox,
                    "State": behavior_state,
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Body detection
    # ------------------------------------------------------------------
    def detect_bodies(
        self, frame: np.ndarray, resize_scale: float = 0.45
    ) -> list[dict]:
        scale = max(0.25, min(1.0, resize_scale))
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rects, _ = self.body_detector.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detections: list[dict] = []
        scale_back = 1.0 / scale
        for x, y, w, h in rects:
            x1 = int(x * scale_back)
            y1 = int(y * scale_back)
            x2 = int((x + w) * scale_back)
            y2 = int((y + h) * scale_back)
            if (x2 - x1) < 60 or (y2 - y1) < 120:
                continue
            detections.append(
                {"Top": y1, "Right": x2, "Bottom": y2, "Left": x1}
            )
        return detections


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def state_for_overlay(state: str) -> str:
    _MAP = {
        "ตั้งใจเรียน": "attentive",
        "ไม่ตั้งใจเรียน": "inattentive",
        "หลับ/เหม่อ": "drowsy",
        "ไม่ทราบสถานะ": "unknown",
    }
    return _MAP.get(state, "other")


def bbox_center(bb: dict) -> tuple[int, int]:
    return ((bb["Left"] + bb["Right"]) // 2, (bb["Top"] + bb["Bottom"]) // 2)


def bbox_iou(a: dict, b: dict) -> float:
    x_left = max(a["Left"], b["Left"])
    y_top = max(a["Top"], b["Top"])
    x_right = min(a["Right"], b["Right"])
    y_bottom = min(a["Bottom"], b["Bottom"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float((a["Right"] - a["Left"]) * (a["Bottom"] - a["Top"]))
    area_b = float((b["Right"] - b["Left"]) * (b["Bottom"] - b["Top"]))
    denom = area_a + area_b - inter_area
    if denom <= 1e-6:
        return 0.0
    return inter_area / denom


def _open_camera_from_env() -> tuple:
    """Open camera from CFRS_CAMERA_SOURCE env var with validation."""
    camera_source_raw = str(os.getenv("CFRS_CAMERA_SOURCE", "0")).strip()
    if not camera_source_raw:
        camera_source_raw = "0"

    camera_source_used = camera_source_raw
    cap = None

    if camera_source_raw.lstrip("-").isdigit():
        camera_index = int(camera_source_raw)
        if camera_index < 0 or camera_index > 10:
            raise ValueError(
                f"Camera index {camera_index} out of reasonable range (0-10)."
            )
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        camera_source_used = str(camera_index)
    else:
        # Validate URL to prevent SSRF
        _validate_url(camera_source_raw, allowed_schemes=("http", "https", "rtsp"))
        cap = cv2.VideoCapture(camera_source_raw)

        # IP Webcam fallback: append /video if base URL doesn't end with it
        if (
            (cap is None or not cap.isOpened())
            and camera_source_raw.startswith(("http://", "https://"))
            and not camera_source_raw.rstrip("/").lower().endswith("/video")
        ):
            fallback_url = f"{camera_source_raw.rstrip('/')}/video"
            cap = cv2.VideoCapture(fallback_url)
            camera_source_used = fallback_url

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera source '{camera_source_raw}'. "
            "Set CFRS_CAMERA_SOURCE=0 for USB webcam or use IP Webcam URL "
            "(example: http://PHONE_IP:8080/video)."
        )

    return cap, camera_source_used


def _send_payload_async(
    backend_url: str, payload: dict, timeout: float = 1.8
) -> None:
    """Send payload to backend in a daemon thread to avoid blocking the main loop."""
    def _post():
        try:
            requests.post(backend_url, json=payload, timeout=timeout)
        except Exception as exc:
            # Logged at caller level with throttle; just silently fail here.
            logger.debug("Async POST failed: %s", exc)

    t = threading.Thread(target=_post, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    service = ClassroomFacialRecognitionService()

    try:
        cap, camera_source_used = _open_camera_from_env()
        logger.info("Camera source = %s", camera_source_used)
    except Exception as exc:
        logger.error("Camera init failed: %s", exc)
        raise SystemExit(1)

    # --- Configuration from env (all clamped to safe ranges) ---
    camera_width = int(os.getenv("CFRS_CAMERA_WIDTH", "960"))
    camera_height = int(os.getenv("CFRS_CAMERA_HEIGHT", "540"))
    frame_resize_scale = max(
        0.35, min(1.0, float(os.getenv("CFRS_FRAME_RESIZE_SCALE", "0.5")))
    )
    process_every_n_frames = max(
        1, min(6, int(os.getenv("CFRS_PROCESS_EVERY_N_FRAMES", "3")))
    )
    body_detect_every_n_frames = max(
        1, min(8, int(os.getenv("CFRS_BODY_DETECT_EVERY_N_FRAMES", "4")))
    )
    body_resize_scale = max(
        0.25, min(0.8, float(os.getenv("CFRS_BODY_RESIZE_SCALE", "0.45")))
    )
    body_match_max_distance = int(
        os.getenv("CFRS_BODY_MATCH_MAX_DISTANCE", "190")
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    # --- Backend / dashboard config ---
    backend_url_raw = os.getenv(
        "CFRS_BACKEND_INGEST_URL", "http://127.0.0.1:5000/api/result"
    )
    try:
        backend_url = _validate_url(backend_url_raw)
    except ValueError as exc:
        logger.error("Invalid backend URL: %s", exc)
        raise SystemExit(1)

    frame_output_path = os.getenv(
        "CFRS_CAMERA_FRAME_PATH", "storage/latest_frame.jpg"
    )
    # Sanitize frame output path
    try:
        frame_output_path = _sanitize_path(frame_output_path)
    except ValueError as exc:
        logger.error("Invalid frame output path: %s", exc)
        raise SystemExit(1)

    post_interval_sec = float(os.getenv("CFRS_POST_INTERVAL_SEC", "1.5"))
    frame_write_interval_sec = float(
        os.getenv("CFRS_FRAME_WRITE_INTERVAL_SEC", "0.45")
    )

    # --- Timing state ---
    last_post_time = 0.0
    last_post_error_time = 0.0
    last_frame_write_time = 0.0
    last_frame_error_time = 0.0
    last_fps_time = time.time()
    smoothed_fps = 0.0
    frame_index = 0
    cached_results: list[dict] = []
    cached_body_boxes: list[dict] = []
    consecutive_read_failures = 0
    MAX_CONSECUTIVE_READ_FAILURES = 30  # ~1 second at 30 FPS

    frame_output_dir = os.path.dirname(frame_output_path) or "."
    os.makedirs(frame_output_dir, exist_ok=True)

    # --- Tracker state ---
    tracked_faces: dict[int, dict] = {}
    next_track_id = 0
    confirmed_names_db: set[str] = set()
    tracker_lock = threading.Lock()

    CONFIRMATION_TIME = 5.0
    TIMEOUT = 10.0
    MAX_DISTANCE = 50

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_read_failures += 1
                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    logger.error(
                        "Lost camera feed after %d consecutive failures. Exiting.",
                        consecutive_read_failures,
                    )
                    break
                logger.debug(
                    "Frame read failed (%d/%d). Retrying...",
                    consecutive_read_failures,
                    MAX_CONSECUTIVE_READ_FAILURES,
                )
                time.sleep(0.03)  # Brief pause before retry
                continue

            consecutive_read_failures = 0  # Reset on success
            current_time = time.time()
            frame_index += 1

            # --- Periodic heavy processing ---
            try:
                should_run_heavy = (
                    frame_index % process_every_n_frames == 1
                ) or (not cached_results)
                if should_run_heavy:
                    cached_results = service.process_frame(
                        frame, resize_scale=frame_resize_scale
                    )
            except Exception as exc:
                logger.error("Face processing error: %s", exc)
                cached_results = []

            results = cached_results

            try:
                should_run_body = (
                    frame_index % body_detect_every_n_frames == 1
                ) or (not cached_body_boxes)
                if should_run_body:
                    cached_body_boxes = service.detect_bodies(
                        frame, resize_scale=body_resize_scale
                    )
            except Exception as exc:
                logger.error("Body detection error: %s", exc)
                cached_body_boxes = []

            body_boxes = cached_body_boxes

            # --- Tracking & rendering ---
            payload_students: list[dict] = []
            drowsy_count = 0
            attentive_count = 0
            inattentive_count = 0
            unknown_count = 0
            seen_track_ids: set[int] = set()
            face_boxes: list[dict] = []

            with tracker_lock:
                # Prune stale tracks
                keys_to_delete = [
                    k
                    for k, v in tracked_faces.items()
                    if current_time - v["last_seen"] > TIMEOUT
                ]
                for k in keys_to_delete:
                    deleted_name = tracked_faces[k].get("name")
                    del tracked_faces[k]
                    # FIX: Allow re-detection after track expires
                    confirmed_names_db.discard(deleted_name)

                for res in results:
                    bb = res["BoundingBox"]
                    ai_name = res["Name"]
                    conf = res["Confidence"]
                    behavior_state = res.get("State", "ไม่ทราบสถานะ")
                    behavior_display = state_for_overlay(behavior_state)
                    face_boxes.append(bb)

                    cx = (bb["Left"] + bb["Right"]) // 2
                    cy = (bb["Top"] + bb["Bottom"]) // 2

                    best_match_id = None
                    min_dist = MAX_DISTANCE

                    for t_id, t_data in tracked_faces.items():
                        dist = math.hypot(
                            cx - t_data["centroid"][0],
                            cy - t_data["centroid"][1],
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_match_id = t_id

                    is_tracking = False
                    is_confirmed = False
                    display_name = ai_name
                    elapsed_time = 0.0
                    payload_track_id = None

                    if best_match_id is None:
                        if ai_name not in ("Unknown", "Moving/Blur"):
                            payload_track_id = next_track_id
                            tracked_faces[next_track_id] = {
                                "name": ai_name,
                                "centroid": (cx, cy),
                                "first_seen": current_time,
                                "last_seen": current_time,
                                "confirmed": False,
                            }
                            display_name = ai_name
                            is_tracking = True
                            next_track_id += 1
                            # FIX: Wrap around to prevent unbounded growth
                            if next_track_id > _MAX_TRACK_ID:
                                next_track_id = 0
                    else:
                        payload_track_id = best_match_id
                        t_data = tracked_faces[best_match_id]
                        t_data["centroid"] = (cx, cy)
                        t_data["last_seen"] = current_time
                        if not t_data["confirmed"] and ai_name not in (
                            "Unknown",
                            "Moving/Blur",
                        ):
                            t_data["name"] = ai_name

                        elapsed_time = current_time - t_data["first_seen"]
                        if elapsed_time >= CONFIRMATION_TIME:
                            t_data["confirmed"] = True
                            display_name = t_data["name"]
                            is_confirmed = True
                            is_tracking = True
                        else:
                            display_name = t_data["name"]
                            is_tracking = True

                    if not is_tracking:
                        if display_name == "Moving/Blur":
                            color = (0, 165, 255)
                            text = "Moving/Blur"
                        else:
                            color = (0, 0, 255)
                            text = f"Unknown ({behavior_display})"
                    else:
                        if is_confirmed:
                            color = (0, 255, 0)
                            text = f"{display_name} ({behavior_display}) {conf}%"
                            if display_name not in confirmed_names_db:
                                logger.info(
                                    ">>> [API/Database] เช็คชื่อ: %s เวลา: %s",
                                    display_name,
                                    datetime.now(),
                                )
                                confirmed_names_db.add(display_name)
                        else:
                            color = (0, 255, 255)
                            countdown = max(
                                0, CONFIRMATION_TIME - elapsed_time
                            )
                            text = (
                                f"Verifying {display_name} "
                                f"({behavior_display}) {countdown:.1f}s"
                            )

                    payload_state = (
                        behavior_state
                        if display_name != "Moving/Blur"
                        else "ไม่ทราบสถานะ"
                    )
                    payload_name = display_name if display_name else "Unknown"
                    payload_students.append(
                        {
                            "track_id": payload_track_id,
                            "name": payload_name,
                            "state": payload_state,
                            "confirmed": bool(is_confirmed),
                            "confidence": float(conf),
                        }
                    )
                    if payload_track_id is not None:
                        seen_track_ids.add(payload_track_id)

                    if payload_state == "หลับ/เหม่อ":
                        drowsy_count += 1
                    elif payload_state == "ตั้งใจเรียน":
                        attentive_count += 1
                    elif payload_state == "ไม่ตั้งใจเรียน":
                        inattentive_count += 1
                    else:
                        unknown_count += 1

                    cv2.rectangle(
                        frame,
                        (bb["Left"], bb["Top"]),
                        (bb["Right"], bb["Bottom"]),
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        text,
                        (bb["Left"], bb["Top"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                # --- Body fallback ---
                for body_bb in body_boxes:
                    overlap_with_face = any(
                        bbox_iou(body_bb, fb) > 0.22 for fb in face_boxes
                    )
                    if overlap_with_face:
                        continue

                    body_cx, body_cy = bbox_center(body_bb)
                    best_track_id = None
                    best_dist = float("inf")

                    for t_id, t_data in tracked_faces.items():
                        if t_id in seen_track_ids:
                            continue
                        dist = math.hypot(
                            body_cx - t_data["centroid"][0],
                            body_cy - t_data["centroid"][1],
                        )
                        if dist < best_dist and dist <= body_match_max_distance:
                            best_dist = dist
                            best_track_id = t_id

                    if best_track_id is None:
                        continue

                    t_data = tracked_faces[best_track_id]
                    t_data["centroid"] = (body_cx, body_cy)
                    t_data["last_seen"] = current_time
                    seen_track_ids.add(best_track_id)

                    inherited_name = t_data.get("name", "Unknown") or "Unknown"
                    if inherited_name in ("Moving/Blur", ""):
                        inherited_name = "Unknown"

                    payload_students.append(
                        {
                            "track_id": best_track_id,
                            "name": inherited_name,
                            "state": "ฟุบหลับ/หันหลัง",
                            "confirmed": False,
                            "confidence": 0.0,
                        }
                    )
                    drowsy_count += 1

                    body_text_name = (
                        inherited_name
                        if inherited_name != "Unknown"
                        else "Unknown"
                    )
                    body_text = f"{body_text_name} (body-fallback drowsy)"
                    cv2.rectangle(
                        frame,
                        (body_bb["Left"], body_bb["Top"]),
                        (body_bb["Right"], body_bb["Bottom"]),
                        (255, 160, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        body_text,
                        (body_bb["Left"], max(18, body_bb["Top"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 180, 50),
                        2,
                    )

            # --- FPS computation ---
            now = time.time()
            dt = max(1e-6, now - last_fps_time)
            last_fps_time = now
            current_fps = 1.0 / dt
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0.0
                else (smoothed_fps * 0.9 + current_fps * 0.1)
            )

            # --- HUD overlay ---
            # FIX: Show total count (faces + body-only) instead of just faces
            total_people = len(results) + sum(
                1
                for s in payload_students
                if s.get("state") == "ฟุบหลับ/หันหลัง"
            )
            cv2.putText(
                frame,
                f"People: {total_people}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (60, 220, 60),
                2,
            )
            cv2.putText(
                frame,
                f"Attentive: {attentive_count}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (80, 210, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Inattentive: {inattentive_count}",
                (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 180, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Drowsy: {drowsy_count}",
                (10, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 60, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Unknown: {unknown_count}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (10, 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                2,
            )

            # --- Backend POST (non-blocking) ---
            if current_time - last_post_time >= post_interval_sec:
                last_post_time = current_time
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "person_count": len(payload_students),
                    "students": payload_students,
                }
                _send_payload_async(backend_url, payload, timeout=1.8)

            # --- Dashboard frame write ---
            if current_time - last_frame_write_time >= frame_write_interval_sec:
                last_frame_write_time = current_time
                try:
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 78],
                    )
                    if ok:
                        _write_dashboard_frame(
                            frame_output_path, jpeg.tobytes()
                        )
                except Exception as exc:
                    if current_time - last_frame_error_time >= 8:
                        logger.warning(
                            "Cannot write camera frame for dashboard: %s", exc
                        )
                        last_frame_error_time = current_time

            cv2.imshow("Ultimate Classroom Identity (Tracker Lock)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Shutting down...")
    except Exception as exc:
        logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
    finally:
        # FIX: Guarantee resource cleanup even on crash
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released. Goodbye.")
