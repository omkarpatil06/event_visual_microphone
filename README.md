# Visual Microphone: Passive Recovery of sound using Event Camera
Using an event camera to passively recover sound from the vibrations of an object. This repository holds all the scipts written for the porject.
Progress list:
1. Simulator: Convert frames from video camera to event frames

## Event Simulator
```event_simulator.py```: Creates a simulator object to process frame based videos to event frames.
```event_camera```: Provides input arguments and configurations for the simulator.

The event simulator takes an input frames-based video and outputs a event-frame based video. The following steps are currently used for the conversion:

1. Stores all frames retrieved from cv2 video read object.
2. Convert frames to luma frames as a measure for intensity.
3. Find the linear log (lin-log) of the luma frames.
4. Apply a low pass filter on the lin-log frames.
5. Substract the current lin-log frame with the one from previous iteration.
6. Based on the difference, if pixels excess a threshold trigger an event.
7. Record the time-stamp, xy address and polarity of all events into a list.
8. Use the event list to create event frames as a matrix.
9. Use cv2 video write to create an mp4 video from the event frames.

As an example the video of elephants is converted to it's event frame representation bellow:

**Original video:**

https://github.com/omkarpatil06/event_visual_microphone/assets/94877472/73c78763-ee7c-452a-b240-0820ba7ee9c2

**Video from simulator**

https://github.com/omkarpatil06/event_visual_microphone/assets/94877472/377e06d4-18ae-46ce-add6-2a6a24c2c0ea


