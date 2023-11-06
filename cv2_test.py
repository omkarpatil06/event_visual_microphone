import cv2 
import numpy as np
import math

CAP_PROP_FPS = 5
CAP_PROP_FRAME_COUNT = 7
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4

channels = 1
frame_width = 1280
frame_height = 720
fps, vid_duration = 30, 10

total_frames = math.floor(fps*vid_duration)
frame_interval = math.floor(1000/fps)
min_range, max_range = 0, 255
mean, std_dev = max_range/2, max_range/4

video_write = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'H264'), fps, [frame_height, frame_width], isColor=False)

for _ in range(total_frames):
    gaussian_frame = np.random.normal(mean, std_dev, (frame_height, frame_width))
    print(gaussian_frame.shape)
    gaussian_frame = np.clip(gaussian_frame, min_range, max_range).astype(np.uint8)
    video_write.write(gaussian_frame)
    
video_write.release()

# video = cv2.VideoCapture('output.mp4')
# if not video.isOpened():
#     print("Error opening video file!")
# else:
#     fps = video.get(CAP_PROP_FPS)
#     frame_count = video.get(CAP_PROP_FRAME_COUNT)
#     frame_width = video.get(CAP_PROP_FRAME_WIDTH)
#     frame_height = video.get(CAP_PROP_FRAME_HEIGHT)
#     print(f"Video frames per second: {fps} fps.")
#     print(f"Video frame count: {frame_count}.")
#     print(f"Video frame size: {frame_width}x{frame_height}.")

# while(video.isOpened()):
#     ret, frame = video.read()
#     #frame = np.asarray(frame)
#     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#     #ret1, frame = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
#     if ret == True:
#         cv2.imshow('Frame', frame)
#         key = cv2.waitKey(frame_interval)
#         if key == ord('q'):
#             break
#     else:
#         break

# video.release()
cv2.destroyAllWindows()