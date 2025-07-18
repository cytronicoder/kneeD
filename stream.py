import time
import queue

import cv2
import numpy as np
import mediapipe as mp

MODEL_PATH = "model/pose_landmarker_full.task"
WINDOW_NAME = "Fast Pose"
SKIP_FRAMES = 0
QUEUE_MAXSIZE = 2
TARGET_FPS = 60
VISIBILITY_THRESHOLD = 0.5

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_HEIGHT = 30
MARGIN = 50

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

disp_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

text_cache = {}
TARGET_MARKERS = {
    23: "L_Hip",
    24: "R_Hip",
    25: "L_Knee",
    26: "R_Knee",
    27: "L_Ankle",
    28: "R_Ankle",
}
LEG_CONNECTIONS = [
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (23, 24),
]


def draw_log(img, fps):
    _cv2_rectangle = cv2.rectangle
    _cv2_putText = cv2.putText
    _cv2_getTextSize = cv2.getTextSize

    h, w, _ = img.shape
    lines = [f"FPS: {fps:.1f}", f"Resolution: {w}x{h}"]
    for i, text in enumerate(lines):
        if text not in text_cache:
            (tw, th), _ = _cv2_getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
            text_cache[text] = (tw, th)
        else:
            tw, th = text_cache[text]

        x = w - tw - MARGIN
        y = MARGIN + (i + 1) * LINE_HEIGHT
        _cv2_rectangle(
            img, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), cv2.FILLED
        )
        _cv2_putText(
            img,
            text,
            (x, y),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )


def draw_markers_readout(img, pose):
    """
    Draw the 2D coordinates and visibility values of the six pose markers
    (L/R Hip, Knee, Ankle) on the output image.

    Args:
        img: The output image to draw on.
        pose: The pose data to draw from.
    """
    _cv2_rectangle = cv2.rectangle
    _cv2_putText = cv2.putText
    _cv2_getTextSize = cv2.getTextSize

    h, w, _ = img.shape

    line_idx = 0
    for i in (23, 24, 25, 26, 27, 28):
        if i >= len(pose):
            continue

        lm = pose[i]
        if lm.visibility < VISIBILITY_THRESHOLD:
            continue

        x_norm, y_norm = lm.x, lm.y
        x_px, y_px = int(x_norm * w), int(y_norm * h)
        visibility = lm.visibility

        text = f"{TARGET_MARKERS[i]}: ({x_norm:.3f},{y_norm:.3f}) [{x_px},{y_px}] vis:{visibility:.2f}"
        y_pos = MARGIN + line_idx * LINE_HEIGHT

        if text not in text_cache:
            (text_width, text_height), _ = _cv2_getTextSize(
                text, FONT, FONT_SCALE, FONT_THICKNESS
            )
            text_cache[text] = (text_width, text_height)
        else:
            text_width, text_height = text_cache[text]

        _cv2_rectangle(
            img,
            (MARGIN - 5, y_pos - text_height - 5),
            (MARGIN + text_width + 5, y_pos + 5),
            (0, 0, 0),
            cv2.FILLED,
        )
        _cv2_putText(
            img,
            text,
            (MARGIN, y_pos),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )
        line_idx += 1


def draw_pose_annotations(frame, pose):
    """Draw pose landmarks and connections on the frame"""
    if not pose:
        return frame

    _cv2_circle = cv2.circle
    _cv2_polylines = cv2.polylines

    h, w, _ = frame.shape

    for i in (23, 24, 25, 26, 27, 28):
        if i < len(pose):
            lm = pose[i]
            x, y = int(lm.x * w), int(lm.y * h)
            _cv2_circle(frame, (x, y), 8, (0, 0, 255), -1)

    lines_to_draw = []
    for si, ei in LEG_CONNECTIONS:
        if si < len(pose) and ei < len(pose):
            x1, y1 = int(pose[si].x * w), int(pose[si].y * h)
            x2, y2 = int(pose[ei].x * w), int(pose[ei].y * h)
            lines_to_draw.append([(x1, y1), (x2, y2)])

    if lines_to_draw:
        lines_array = np.array(lines_to_draw, dtype=np.int32)
        _cv2_polylines(frame, lines_array, False, (0, 255, 0), 6)

    return frame


def on_result(result, mp_image: mp.Image, timestamp_ms: int):
    frame = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
    pose = result.pose_landmarks[0] if result.pose_landmarks else None

    try:
        if disp_queue.full():
            disp_queue.get_nowait()
        disp_queue.put_nowait((frame, pose, time.monotonic()))
    except queue.Full:
        pass


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=on_result,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    frame_idx = 0
    last_display_ts = time.monotonic()
    last_annotated = None
    last_pose = None
    last_fps = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        if frame_idx % (SKIP_FRAMES + 1) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, int(time.monotonic() * 1000))
        try:
            frame_data, pose, ts = disp_queue.get_nowait()
            annotated_frame = draw_pose_annotations(frame_data.copy(), pose)
            last_annotated = annotated_frame
            last_pose = pose
            now = time.monotonic()
            last_fps = 1.0 / (now - last_display_ts) if now > last_display_ts else 0.0
            last_display_ts = now
        except queue.Empty:
            pass
        if last_annotated is not None:
            display = last_annotated.copy()
            draw_log(display, last_fps)
            if last_pose:
                draw_markers_readout(display, last_pose)

            display = cv2.resize(display, (1280, 960), interpolation=cv2.INTER_LINEAR)
        else:
            display = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
