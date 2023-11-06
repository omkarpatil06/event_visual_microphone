import cv2
import torch
import math
import numpy as np
import torch.nn.functional as F

class EventSimulator():
    def __init__(self, input_file, output_file, positive_threshold, negative_threshold, cutoff_freq, full_scale_count):
        self.input_file = input_file
        self.output_file = output_file
        self.pos_thresh = positive_threshold
        self.neg_thresh = negative_threshold
        self.cutoff_freq = cutoff_freq
        self.full_scale_count = float(full_scale_count)

        self.luma_frame = None
        self.lin_log_frame = None
        self.intensity_frame = None
        self.previous_lp_log_frame = None
        self.current_lp_log_frame = None
        self.difference_frame = None

        self.previous_frame_time = None
        self.current_frame_time = None
        self.events = torch.empty((0, 4), dtype=torch.float32)
        self.render_previous_time = None
        self.render_current_time = None
        self.render_next_time = None
        self.event_segment = None
        self.current_frame = None
        self.final_events_frame = None
        
    def event_simulator(self):
        # cv2 video read object to read video frames
        video_read = cv2.VideoCapture(self.input_file)

        # cv2 retrieving meta-data
        self.delta_time = 1/video_read.get(cv2.CAP_PROP_FPS)
        self.frame_counts = int(video_read.get(cv2.CAP_PROP_FRAME_COUNT))-600
        self.width = int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_interval = int(1000*self.delta_time)
        self.duration = float(self.frame_counts)/self.delta_time
        
        # store all frames from the input video in frames
        frames = [] 
        while True:
            ret, frame = video_read.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = torch.tensor(frame, dtype=torch.float64)
            frames.append(frame)
        # Convert the list of frames to a PyTorch tensor
        frames = torch.stack(frames)
        print(f"Completed retrieving frames of size {frames.shape}!")
        
        # process each frame based on the block diagram from v2e emulator by Delbruck et.al.
        for i in range(self.frame_counts):
            # updating time variables
            if self.previous_frame_time is None:
                self.previous_frame_time = 0
            if self.current_frame_time is None:
                self.current_frame_time = 0
            self.current_frame_time += self.delta_time

            # frames going through diffrent blocks
            self.luma_frame = frames[i]
            self.lin_log_frame = self.lin_log()
            self.intensity_frame = self.normalise_intensity_frame()
            if self.previous_lp_log_frame is None:
                self.previous_lp_log_frame = self.lin_log_frame
            self.current_lp_log_frame = self.low_pass_filter()
            self.difference_frame = self.previous_lp_log_frame - self.current_lp_log_frame

            # convert all events to a Nx4 array of events
            self.assemble_events()

            # updating time variables
            self.previous_frame_time = self.current_frame_time
            self.previous_lp_log_frame = self.current_lp_log_frame
        print(f"Completed event list with {self.events.shape} events!")

    def render_video(self):
        # convert event list to a list of frames [M, height, width]
        self.final_events_frame = self.event_to_frame()

        # cv2 object for writing frames to a video
        print(f"Completed event list to event frames conversion with size {self.final_events_frame.shape}!")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_write = cv2.VideoWriter(self.output_file, fourcc, 1/self.delta_time, [self.height, self.height], isColor=False)

        # correction of frame to correct orientation and writing the frames to video output file
        for i in range(self.frame_counts):
            self.final_events_frame[i] = np.rot90(self.final_events_frame[i], k=3)
            self.final_events_frame[i] = np.fliplr(self.final_events_frame[i])
            video_write.write(self.final_events_frame[i])
        
        # procedure after completing all event frames
        print("Completed rendering event frames!")
        video_write.release()
        cv2.destroyAllWindows()
    
    def lin_log(self):
        # luma_frame: is a luma frame of size [height, width]
        # threshold: a value from 0-255, generally low, to prevent low dynamic range
        
        threshold=20.0
        linearisation = (1.0/threshold)*math.log(threshold)
        
        # if pixel has value less than threshold apply the linear gradient, else find its log.
        lin_log_frame = torch.where(self.luma_frame <= threshold, self.luma_frame*linearisation, torch.log(self.luma_frame))
        lin_log_frame = torch.round(lin_log_frame*1e8)/1e8
        return lin_log_frame.float()    # is a 32-bit float type

    def normalise_intensity_frame(self):
        # luma_frame: is a luma frame of size [height, width]
        return (self.luma_frame+20)/275  # scale from 0.1-1

    def low_pass_filter(self):
        # lin_log_frame: is the lin log frame of the luma frame of size [height, width]
        # previous_lp_log_frame: the low pass filter frame from previous frame
        # intensity_frame: the normalised intensity frame of luma frame
        # delta_time: time difference between current and previous frame
        # cutoff_freq: the cutoff frequency for IIR filter
        
        # if cut-off freq is 0, the filter won't be applied
        if self.cutoff_freq <= 0:
            return self.lin_log_frame
        
        # compute the time-constant tau and then epsilon based on delta_time and rescaling it
        time_constant = 1/(2*math.pi*self.cutoff_freq)
        epsilon = self.intensity_frame*(self.delta_time/time_constant)  
        epsilon = torch.clamp(epsilon, max=1)

        # filter based of a sum of average
        current_lp_log_frame = (1-epsilon)*self.previous_lp_log_frame + epsilon*self.lin_log_frame
        return current_lp_log_frame

    def event_frame(self):
        # difference_frame: lin log frame - previous base lin log frame
        # positive_threshold: log threshold before triggering a positive event
        # negative_threshold: log threshold before triggering a negative event

        # if difference at a pixel is negative replace with 0
        positive_bool_frame = F.relu(self.difference_frame)
        # if difference at a pixel is positive replace with 0
        negative_bool_frame = F.relu(-self.difference_frame)

        # compute how many events are triggered at a pixel based on difference and threshold
        pos_event_frame = torch.div(positive_bool_frame, self.pos_thresh, rounding_mode='floor')
        pos_event_frame = pos_event_frame.type(torch.int32)
        neg_event_frame = torch.div(negative_bool_frame, self.neg_thresh, rounding_mode='floor')
        neg_event_frame = neg_event_frame.type(torch.int32)
        return pos_event_frame, neg_event_frame

    def assemble_events(self):
        # find a bool matrix of all pixels with positive and negative events
        pos_event_frame, neg_event_frame = self.event_frame()

        # finding maximum number of events of any pixel in the frame
        max_number_events = max(pos_event_frame.max(), neg_event_frame.max())
        time_frame_segments = max_number_events if max_number_events > 0 else 1
        time_frame_intervals = self.delta_time/time_frame_segments
        # diving the time between frames by max number of events of any pixel in the frame
        ts = torch.linspace(start=self.previous_frame_time+time_frame_intervals, end=self.current_frame_time, steps=time_frame_segments, dtype=torch.float32)
        current_event_list = None
        if max_number_events == 0:
            return
        else:
            # storing events in Nx4 array based on how many events occur at a pixel
            for i in range(max_number_events):
                pos_xy = (pos_event_frame >= i + 1)
                neg_xy = (neg_event_frame >= i + 1)
                pos_yaddr, pos_xaddr = pos_xy.nonzero(as_tuple=True)
                neg_yaddr, neg_xaddr = neg_xy.nonzero(as_tuple=True)
                current_event_list = self.event_list(pos_yaddr, pos_xaddr, neg_yaddr, neg_xaddr, ts[i])
                self.events = torch.cat((self.events, current_event_list), dim=0)

    def event_list(self, pos_yaddr, pos_xaddr, neg_yaddr, neg_xaddr, event_timestamp):
        total_positive_events = pos_yaddr.shape[0]
        total_negative_events = neg_yaddr.shape[0]
        total_events = total_positive_events + total_negative_events
        current_event_list = None
        if total_events > 0:
            current_event_list = torch.ones((total_events, 4), dtype=torch.float32)
            current_event_list[:, 0] *= event_timestamp
            current_event_list[:total_positive_events, 1] = pos_yaddr
            current_event_list[:total_positive_events, 2] = pos_xaddr
            current_event_list[total_positive_events:, 1] = neg_yaddr
            current_event_list[total_positive_events:, 2] = neg_xaddr
            current_event_list[total_positive_events:, 3] = -1
        return current_event_list

    def accumulate_event_frame(self):
        positive_polarity = (self.event_segment[:, 3] == 1).numpy()
        negative_polarity = np.logical_not(positive_polarity)
        positive_frame, _, _ = np.histogram2d(self.event_segment[positive_polarity, 2], self.event_segment[positive_polarity, 1], bins=(self.width, self.height))
        negative_frame, _, _ = np.histogram2d(self.event_segment[negative_polarity, 2], self.event_segment[negative_polarity, 1], bins=(self.width, self.height))
        
        self.current_frame = np.zeros_like(positive_frame)
        self.current_frame = np.clip(self.current_frame + (positive_frame - negative_frame), -self.full_scale_count, self.full_scale_count).astype(np.uint8)
        self.current_frame = self.current_frame[:self.height, :]
        return self.current_frame

    def event_to_frame(self):
        time = self.events[:,0]
        if self.render_current_time == None:
            self.render_current_time = time[0]
        self.render_next_time = self.render_current_time + self.delta_time
        
        total_events = len(time)
        # inital_dynamic_idx = 0
        start_idx, end_idx = 0, total_events
        complete_event_list = False
        final_event_frame = np.empty((self.frame_counts, self.height, self.height), dtype=np.uint8)
        
        counter = 0
        while not complete_event_list:
            start_idx = np.searchsorted(time, self.render_current_time, side='left')
            end_idx = np.searchsorted(time, self.render_next_time, side='right')
            if end_idx >= total_events - 1:
                complete_event_list = True
                end_idx = total_events - 1
            self.event_segment = self.events[start_idx:end_idx]
            final_event_frame[counter] = 255.*(self.accumulate_event_frame()/self.full_scale_count)
            
            if not complete_event_list:
                self.render_current_time += self.delta_time
                self.render_next_time = self.render_current_time + self.delta_time
                counter += 1
                
        return final_event_frame
