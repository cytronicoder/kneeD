import time
import logging
import signal
import sys
from collections import deque
from threading import Lock
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List, Deque

import cv2
import numpy as np
import mediapipe as mp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PoseViewerConfig:
    """Configuration parameters for the PoseViewer application."""

    display_width: int = 1280
    display_height: int = 960
    camera_width: int = 640
    camera_height: int = 480

    marker_color: Tuple[int, int, int] = (0, 0, 255)
    connection_color: Tuple[int, int, int] = (0, 255, 0)
    text_background_color: Tuple[int, int, int] = (0, 0, 0)
    text_color: Tuple[int, int, int] = (255, 255, 255)

    marker_radius: int = 8
    connection_thickness: int = 6
    text_background_padding: int = 2

    font_scale: float = 0.35
    font_thickness: int = 1
    line_height: int = 15
    margin: int = 25

    camera_buffer_size: int = 1
    frame_summary_interval: int = 30
    max_consecutive_errors: int = 10

    fps_smoothing_alpha: float = 0.9
    visibility_threshold: float = 0.5

    draw_full_pose: bool = False


class PoseViewer:
    """
    A real-time pose detection and visualization application using MediaPipe.
    """

    def __init__(
        self,
        model_path: str,
        target_fps: int = 60,
        window_name: str = "Fast Pose",
        queue_maxsize: int = 2,
        config: Optional[PoseViewerConfig] = None,
    ) -> None:
        """
        Initialize the PoseViewer.

        Args:
            model_path: Path to the MediaPipe pose model file
            target_fps: Target frames per second for pose detection
            window_name: Name of the display window
            queue_maxsize: Maximum size of the display queue
            config: PoseViewerConfig instance for all configuration parameters
        """
        self.model_path = model_path
        self.target_fps = target_fps
        self.window_name = window_name
        self.queue_maxsize = queue_maxsize
        self.config = config or PoseViewerConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.disp_queue: Deque[Tuple[np.ndarray, Optional[Any], float]] = deque(
            maxlen=queue_maxsize
        )
        self._queue_lock = Lock()
        self.text_cache: Dict[str, Tuple[int, int]] = {}
        self.target_markers: Dict[int, str] = {
            23: "L_Hip",
            24: "R_Hip",
            25: "L_Knee",
            26: "R_Knee",
            27: "L_Ankle",
            28: "R_Ankle",
        }
        self.leg_connections: List[Tuple[int, int]] = [
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
            (23, 24),
        ]
        self._setup_mediapipe()
        self._setup_window()
        self._setup_signal_handlers()

        logger.info("PoseViewer initialized successfully")

    def _setup_mediapipe(self) -> None:
        """
        Initialize the MediaPipe pose landmarker.

        Sets up the MediaPipe pose landmarker based on the model path
        provided at initialization. The result callback is set to
        _on_result, and the running mode is set to VisionRunningMode.LIVE_STREAM.
        After initialization, the landmarker is stored in the instance
        variable self.landmarker.

        Raises:
            OSError
            ValueError
            RuntimeError
        """
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=self._on_result,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            logger.info("MediaPipe pose landmarker initialized successfully")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to initialize MediaPipe pose landmarker: %s", e)
            raise

    def _setup_window(self) -> None:
        """
        Set up the OpenCV window for displaying the video feed.

        Initializes a named window with normal properties and resizes it
        to the dimensions specified in the configuration.

        Raises:
            cv2.error: If there is an issue creating or resizing the window.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name, self.config.display_width, self.config.display_height
        )

    def _setup_signal_handlers(self) -> None:
        """
        Configure signal handlers for graceful shutdown.

        This method configures the signal handlers for SIGINT, SIGTERM, SIGHUP,
        and SIGQUIT to call the _cleanup method and exit the process when
        received. This allows the application to clean up resources when
        terminated.

        The signal handlers all call _cleanup, which handles releasing the
        camera and destroying any OpenCV windows, before exiting the process.

        This method is called at initialization time.
        """

        def signal_handler(signum: int, frame: Any) -> None:
            """
            Signal handler for SIGINT, SIGTERM, SIGHUP, and SIGQUIT.

            This function is called when one of the above signals is received.
            It initiates a graceful shutdown by calling _cleanup to release
            resources such as the camera and any OpenCV windows, and then exits
            the process using sys.exit(0).

            Args:
                signum: The signal number received
                frame: The current stack frame (unused)
            """
            logger.info("Received signal %s, initiating graceful shutdown", signum)
            self._cleanup()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)
        if hasattr(signal, "SIGQUIT"):
            signal.signal(signal.SIGQUIT, signal_handler)

        logger.debug("Signal handlers configured for graceful shutdown")

    def _draw_text_block(
        self, img: np.ndarray, text: str, pos: Tuple[int, int]
    ) -> None:
        """
        Draw text with a background rectangle at the specified position.

        Args:
            img: The image to draw on.
            text: The text to draw.
            pos: Tuple (x, y) representing the position to draw the text.
        """
        if text not in self.text_cache:
            (tw, th), _ = cv2.getTextSize(
                text, self.font, self.config.font_scale, self.config.font_thickness
            )
            self.text_cache[text] = (tw, th)
        else:
            tw, th = self.text_cache[text]

        x, y = pos
        padding = self.config.text_background_padding
        cv2.rectangle(
            img,
            (x - padding, y - th - padding),
            (x + tw + padding, y + padding),
            self.config.text_background_color,
            cv2.FILLED,
        )
        cv2.putText(
            img,
            text,
            (x, y),
            self.font,
            self.config.font_scale,
            self.config.text_color,
            self.config.font_thickness,
        )

    def _draw_log(self, img: np.ndarray, fps: float) -> None:
        """
        Draw some basic logging information on the frame.

        Args:
            img: The frame to draw on.
            fps: The current frames per second.
        """
        h, w, _ = img.shape
        lines = [f"FPS: {fps:.1f}", f"Resolution: {w}x{h}"]
        for i, text in enumerate(lines):
            if text not in self.text_cache:
                (tw, th), _ = cv2.getTextSize(
                    text, self.font, self.config.font_scale, self.config.font_thickness
                )
                self.text_cache[text] = (tw, th)
            else:
                tw, th = self.text_cache[text]

            x = w - tw - self.config.margin
            y = self.config.margin + i * self.config.line_height
            self._draw_text_block(img, text, (x, y))

    def _draw_markers_readout(self, img: np.ndarray, pose: Any) -> None:
        """
        Draw text blocks with information about the detected pose landmarks.

        Args:
            img: The frame to draw on.
            pose: The pose data to draw from.

        Notes:
            If `draw_full_pose` is `True`, draws information for all detected
            landmarks. Otherwise, draws information only for the subset of
            landmarks in `target_markers`.
        """
        h, w, _ = img.shape
        line_idx = 0

        if self.config.draw_full_pose:
            for idx, lm in enumerate(pose):
                if lm.visibility < self.config.visibility_threshold:
                    continue

                x_norm, y_norm = lm.x, lm.y
                x_px, y_px = int(x_norm * w), int(y_norm * h)
                visibility = lm.visibility

                name = mp.solutions.pose.PoseLandmark(idx).name

                text = f"{name}: ({x_norm:.3f},{y_norm:.3f})  px[{x_px},{y_px}]  vis:{visibility:.2f}"
                y_pos = self.config.margin + line_idx * self.config.line_height

                self._draw_text_block(img, text, (self.config.margin, y_pos))
                line_idx += 1
        else:
            for i, label in self.target_markers.items():
                if i >= len(pose):
                    continue
                lm = pose[i]
                if lm.visibility < self.config.visibility_threshold:
                    continue

                x_norm, y_norm = lm.x, lm.y
                x_px, y_px = int(x_norm * w), int(y_norm * h)
                visibility = lm.visibility

                text = f"{label}: ({x_norm:.3f},{y_norm:.3f})  px[{x_px},{y_px}]  vis:{visibility:.2f}"
                y_pos = self.config.margin + line_idx * self.config.line_height

                self._draw_text_block(img, text, (self.config.margin, y_pos))
                line_idx += 1

    def _draw_pose_annotations(
        self, frame: np.ndarray, pose: Optional[Any]
    ) -> np.ndarray:
        """
        Draw the pose annotations on the given frame.

        Args:
            frame: The frame to draw on.
            pose: The pose data to draw from.

        Returns:
            The annotated frame.
        """
        if not pose:
            return frame

        h, w, _ = frame.shape

        if self.config.draw_full_pose:
            for idx, lm in enumerate(pose):
                if lm.visibility < self.config.visibility_threshold:
                    continue
                x, y = int(lm.x * w), int(lm.y * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(
                        frame,
                        (x, y),
                        self.config.marker_radius,
                        self.config.marker_color,
                        cv2.FILLED,
                    )

            for si, ei in mp.solutions.pose.POSE_CONNECTIONS:
                lm1 = pose[si]
                lm2 = pose[ei]
                if (
                    lm1.visibility < self.config.visibility_threshold
                    or lm2.visibility < self.config.visibility_threshold
                ):
                    continue
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        self.config.connection_color,
                        self.config.connection_thickness,
                    )
            return frame

        return self._draw_pose_annotations_manual(frame, pose)

    def _draw_pose_annotations_manual(
        self, frame: np.ndarray, pose: Optional[Any]
    ) -> np.ndarray:
        """
        Draw pose annotations manually (legs only).

        Args:
            frame: The frame to draw on.
            pose: The pose data to draw from.

        Returns:
            The annotated frame.
        """
        if not pose:
            return frame

        try:
            _cv2_circle = cv2.circle
            _cv2_polylines = cv2.polylines

            h, w, _ = frame.shape

            for i in (23, 24, 25, 26, 27, 28):
                if i < len(pose):
                    try:
                        lm = pose[i]
                        x, y = int(lm.x * w), int(lm.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            _cv2_circle(
                                frame,
                                (x, y),
                                self.config.marker_radius,
                                self.config.marker_color,
                                -1,
                            )
                    except (AttributeError, ValueError, TypeError) as e:
                        logger.debug("Error drawing marker %d: %s", i, e)

            lines_to_draw = []
            for si, ei in self.leg_connections:
                if si < len(pose) and ei < len(pose):
                    try:
                        x1, y1 = int(pose[si].x * w), int(pose[si].y * h)
                        x2, y2 = int(pose[ei].x * w), int(pose[ei].y * h)
                        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                            lines_to_draw.append([(x1, y1), (x2, y2)])
                    except (AttributeError, ValueError, TypeError) as e:
                        logger.debug("Error preparing line %d-%d: %s", si, ei, e)

            if lines_to_draw:
                try:
                    lines_array = np.array(lines_to_draw, dtype=np.int32)
                    _cv2_polylines(
                        frame,
                        lines_array,
                        False,
                        self.config.connection_color,
                        self.config.connection_thickness,
                    )
                except (ValueError, TypeError) as e:
                    logger.debug("Error drawing lines: %s", e)
        except (AttributeError, ValueError, TypeError) as e:
            logger.error("Error in pose annotation drawing: %s", e)

        return frame

    def _on_result(self, result: Any, mp_image: mp.Image, timestamp_ms: int) -> None:
        """
        Callback to handle pose landmark detection results from the pose landmarker.

        Args:
            result: The pose landmark detection result from the pose landmarker.
            mp_image: The Mediapipe image frame that the pose landmarker processed.
            timestamp_ms: The timestamp of the frame in milliseconds.

        This callback takes the results from the pose landmarker and puts them into
        the display queue for the main thread to consume.
        """
        callback_start = time.perf_counter()
        try:
            convert_start = time.perf_counter()
            frame = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
            convert_time = time.perf_counter() - convert_start

            pose = None
            if result.pose_landmarks:
                if len(result.pose_landmarks) > 1:
                    logger.warning(
                        "Multiple people detected (%d), using first person",
                        len(result.pose_landmarks),
                    )
                pose = result.pose_landmarks[0]
                logger.debug("Pose detected with %d landmarks", len(pose))
            else:
                logger.debug("No pose detected in frame")

            queue_start = time.perf_counter()
            try:
                with self._queue_lock:
                    self.disp_queue.append((frame, pose, time.monotonic()))
                queue_time = time.perf_counter() - queue_start

                total_callback_time = time.perf_counter() - callback_start

                logger.debug(
                    "Detection callback: convert=%.1fms, "
                    "queue=%.1fms, "
                    "total=%.1fms, "
                    "queue_len=%d",
                    convert_time * 1000,
                    queue_time * 1000,
                    total_callback_time * 1000,
                    len(self.disp_queue),
                )
            except (OSError, RuntimeError) as e:
                logger.error("Error adding frame to queue: %s", e)

        except (AttributeError, ValueError, TypeError) as e:
            logger.error("Error in pose detection callback: %s", e)
            try:
                if hasattr(mp_image, "numpy_view"):
                    frame = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
                    with self._queue_lock:
                        self.disp_queue.append((frame, None, time.monotonic()))
            except (AttributeError, ValueError, TypeError) as fallback_error:
                logger.error(
                    "Fallback frame processing also failed: %s", fallback_error
                )

    def _cleanup(self, cap: Optional[cv2.VideoCapture] = None) -> None:
        """
        Clean up resources including MediaPipe landmarker, camera, and OpenCV windows.

        Args:
            cap: OpenCV VideoCapture object to release (optional)
        """
        logger.info("Cleaning up resources")

        try:
            if hasattr(self, "landmarker"):
                self.landmarker.close()
                logger.debug("MediaPipe landmarker closed")
        except (AttributeError, RuntimeError) as e:
            logger.error("Error closing landmarker: %s", e)

        if cap is not None:
            try:
                cap.release()
                logger.debug("Camera released")
            except AttributeError as e:
                logger.error("Error releasing camera: %s", e)

        try:
            cv2.destroyAllWindows()
            logger.debug("OpenCV windows destroyed")
        except AttributeError as e:
            logger.error("Error destroying windows: %s", e)

        logger.info("Cleanup completed successfully")

    def _init_camera(self) -> cv2.VideoCapture:
        """
        Initialize the camera with specified properties.

        Returns:
            cv2.VideoCapture: Initialized camera object

        Raises:
            RuntimeError: If camera cannot be opened
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error(
                "Cannot open webcam - check if camera is connected and not in use"
            )
            raise RuntimeError("Cannot open webcam")

        fps_set = cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        width_set = cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        height_set = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        buffer_set = cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera_buffer_size)

        if not all([fps_set, width_set, height_set, buffer_set]):
            logger.warning("Some camera properties could not be set")

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(
            "Camera initialized: %dx%d @ %.1f FPS",
            int(actual_width),
            int(actual_height),
            actual_fps,
        )

        return cap

    def _handle_frame(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        """
        Handle a single frame: run pose detection if needed and process queue.

        Args:
            frame: The camera frame to process
            state: Dictionary containing processing state variables
        """
        now = time.monotonic()
        elapsed = now - state["last_detect_ts"]
        if elapsed >= state["target_detect_interval"]:
            try:
                convert_start = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                convert_time = time.perf_counter() - convert_start

                detect_start = time.perf_counter()
                self.landmarker.detect_async(mp_img, int(now * 1000))
                detect_submit_time = time.perf_counter() - detect_start

                state["last_detect_ts"] = now

                logger.debug(
                    "Frame processing: convert=%.1fms, " "detect_submit=%.1fms",
                    convert_time * 1000,
                    detect_submit_time * 1000,
                )
            except (ValueError, TypeError, RuntimeError) as e:
                logger.error("Error in pose detection: %s", e)

        queue_start = time.perf_counter()

        try:
            with self._queue_lock:
                if self.disp_queue:
                    frame_data, pose, ts = self.disp_queue.popleft()
                else:
                    frame_data = pose = ts = None
            queue_time = time.perf_counter() - queue_start

            if frame_data is not None:
                annotate_start = time.perf_counter()
                annotated_frame = self._draw_pose_annotations(frame_data, pose)
                annotate_time = time.perf_counter() - annotate_start

                text_start = time.perf_counter()
                self._draw_log(annotated_frame, state["last_fps"])
                if pose:
                    self._draw_markers_readout(annotated_frame, pose)
                text_time = time.perf_counter() - text_start

                state["last_annotated"] = annotated_frame
                state["last_pose"] = pose

                now = time.monotonic()
                if now > state["last_display_ts"]:
                    instant_fps = 1.0 / (now - state["last_display_ts"])
                    if state["last_fps"] == 0.0:
                        state["last_fps"] = instant_fps
                    else:
                        state["last_fps"] = (
                            self.config.fps_smoothing_alpha * state["last_fps"]
                            + (1 - self.config.fps_smoothing_alpha) * instant_fps
                        )
                state["last_display_ts"] = now

                logger.debug(
                    "Frame rendering: queue=%.1fms, " "annotate=%.1fms, " "text=%.1fms",
                    queue_time * 1000,
                    annotate_time * 1000,
                    text_time * 1000,
                )
            else:
                logger.debug("Queue access: %.1fms (empty)", queue_time * 1000)
        except IndexError:
            queue_time = time.perf_counter() - queue_start
            logger.debug("Queue access: %.1fms (index error)", queue_time * 1000)
        except (OSError, RuntimeError) as e:
            logger.error("Error processing display queue: %s", e)

    def _draw_and_show(self, frame: np.ndarray, state: Dict[str, Any]) -> bool:
        """
        Draw the final display and show it in the window.

        Args:
            frame: The current camera frame
            state: Dictionary containing processing state variables

        Returns:
            bool: True if should continue, False if should exit
        """
        display_start = time.perf_counter()
        try:
            prep_start = time.perf_counter()
            if state["last_annotated"] is not None:
                display = cv2.resize(
                    state["last_annotated"],
                    (self.config.display_width, self.config.display_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                display = cv2.resize(
                    frame,
                    (self.config.display_width, self.config.display_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            prep_time = time.perf_counter() - prep_start

            show_start = time.perf_counter()
            cv2.imshow(self.window_name, display)
            show_time = time.perf_counter() - show_start

            key_start = time.perf_counter()
            key_result = cv2.waitKey(1) & 0xFF == 27
            key_time = time.perf_counter() - key_start

            total_display_time = time.perf_counter() - display_start

            logger.debug(
                "Display pipeline: prep=%.1fms, "
                "show=%.1fms, "
                "key=%.1fms, "
                "total=%.1fms",
                prep_time * 1000,
                show_time * 1000,
                key_time * 1000,
                total_display_time * 1000,
            )

            if key_result:
                logger.info("ESC key pressed, exiting")
                return False
            return True
        except (AttributeError, TypeError) as e:
            logger.error("Error in display operations: %s", e)
            return True

    def run(self) -> None:
        """
        Main execution loop for the pose detection application.

        Initializes the camera and starts the main processing loop.
        The application exits when the user presses the 'Esc' key.
        """
        logger.info("Starting pose detection application")

        try:
            cap = self._init_camera()

            state: Dict[str, Any] = {
                "last_display_ts": time.monotonic(),
                "last_detect_ts": 0.0,
                "target_detect_interval": 1.0 / self.target_fps,
                "last_annotated": None,
                "last_pose": None,
                "last_fps": 0.0,
                "error_count": 0,
                "max_errors": self.config.max_consecutive_errors,
                "frame_idx": 0,
            }

            logger.info("Starting main processing loop")

            try:
                while True:
                    try:
                        success, frame = cap.read()
                        if not success:
                            logger.warning("Failed to read frame from camera")
                            state["error_count"] += 1
                            if state["error_count"] > state["max_errors"]:
                                logger.error(
                                    "Too many consecutive frame read errors, exiting"
                                )
                                break
                            continue

                        state["error_count"] = 0
                        state["frame_idx"] += 1

                        frame_start = time.perf_counter()
                        self._handle_frame(frame, state)
                        frame_process_time = time.perf_counter() - frame_start

                        display_start = time.perf_counter()
                        should_continue = self._draw_and_show(frame, state)
                        display_time = time.perf_counter() - display_start

                        if state["frame_idx"] % self.config.frame_summary_interval == 0:
                            total_frame_time = frame_process_time + display_time
                            logger.info(
                                "Frame %d: "
                                "process=%.1fms, "
                                "display=%.1fms, "
                                "total=%.1fms, "
                                "fps=%.1f",
                                state["frame_idx"],
                                frame_process_time * 1000,
                                display_time * 1000,
                                total_frame_time * 1000,
                                state["last_fps"],
                            )

                        if not should_continue:
                            break
                    except KeyboardInterrupt:
                        logger.info("Keyboard interrupt received, exiting")
                        break
                    except (RuntimeError, OSError, ValueError) as e:
                        logger.error("Unexpected error in main loop: %s", e)
                        state["error_count"] += 1
                        if state["error_count"] > state["max_errors"]:
                            logger.error("Too many consecutive errors, exiting")
                            break
            finally:
                self._cleanup(cap)
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Fatal error in main function: %s", e)
            raise
        finally:
            self._cleanup()
