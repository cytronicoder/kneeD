from PoseViewer import PoseViewer, PoseViewerConfig

model_path = "model/pose_landmarker_full.task"
config = PoseViewerConfig(
    draw_full_pose=True,
)
viewer = PoseViewer(model_path=model_path, config=config)
viewer.run()
