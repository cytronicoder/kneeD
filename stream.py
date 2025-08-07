"""
Client code. This script sets up a command-line interface for running the PoseViewer application.
It allows users to specify camera sources and whether to draw full pose landmarks or just leg landmarks.
"""

import argparse
from PoseViewer import PoseViewer, PoseViewerConfig


def main():
    """
    Main function to run the PoseViewer application.
    """
    parser = argparse.ArgumentParser(
        description="Real-time pose detection with MediaPipe"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera source: device index (0, 1, 2...) or URL (e.g., http://192.168.1.42:8080/video)",
    )
    parser.add_argument(
        "--full-pose", action="store_true", help="Draw full pose instead of just legs"
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Hide information overlays (FPS, coordinates)",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Enable data collection for biomechanical analysis",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="pose_data.csv",
        help="Output CSV file for data collection (default: pose_data.csv)",
    )
    parser.add_argument(
        "--detect-calibration",
        action="store_true",
        help="Enable automatic calibration grid detection (A4 paper with 1cm squares)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target FPS for data collection (default: 60, set to 120 for research)",
    )
    args = parser.parse_args()

    model_path = "model/pose_landmarker_full.task"
    config = PoseViewerConfig(
        draw_full_pose=args.full_pose,
        camera_source=args.source,
        show_info_overlay=not args.no_info,
        save_data=args.save_data,
        output_file=args.output_file,
        detect_calibration=args.detect_calibration,
    )

    print(f"Starting pose detection with camera source: {args.source}")
    if args.full_pose:
        print("Drawing full pose landmarks")
    else:
        print("Drawing leg landmarks only")

    if args.no_info:
        print("Information overlays disabled")
    else:
        print("Information overlays enabled (FPS, coordinates)")

    if args.save_data:
        print(f"Data collection enabled → {args.output_file}")
        if args.detect_calibration:
            print("Calibration grid detection enabled")
        print(f"Target FPS: {args.fps}")

    viewer = PoseViewer(model_path=model_path, target_fps=args.fps, config=config)
    viewer.run()


if __name__ == "__main__":
    main()
