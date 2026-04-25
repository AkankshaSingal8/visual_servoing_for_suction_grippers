import pyzed.sl as sl
import cv2

input_path = "dinov2_track_20260407_030013.svo2"
output_path = "output.mp4"

zed = sl.Camera()
init = sl.InitParameters()
init.set_from_svo_file(input_path)
init.svo_real_time_mode = False  # faster than real-time playback

zed.open(init)

runtime = sl.RuntimeParameters()
mat = sl.Mat()

# get resolution dynamically (important)
camera_info = zed.get_camera_information()
res = camera_info.camera_configuration.resolution
width, height = res.width, res.height

fps = 30  # or extract from metadata if needed

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        break

    zed.retrieve_image(mat, sl.VIEW.LEFT)
    frame = mat.get_data()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    out.write(frame)

out.release()
zed.close()