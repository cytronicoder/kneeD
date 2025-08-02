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
    args = parser.parse_args()

    model_path = "model/pose_landmarker_full.task"
    config = PoseViewerConfig(
        draw_full_pose=args.full_pose,
        camera_source=args.source,
        show_info_overlay=not args.no_info,  # Invert since --no-info hides overlays
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

    viewer = PoseViewer(model_path=model_path, config=config)
    viewer.run()


if __name__ == "__main__":
    main()
