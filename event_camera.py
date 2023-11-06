from event_simulator import EventSimulator

input_file = 'elefant.mp4'
output_file = 'event.mp4'
positive_threshold = 0.15
negative_threshold = 0.15
cutoff_freq = 3000
full_scale_count = 3

vid = EventSimulator(input_file=input_file, output_file=output_file, positive_threshold=positive_threshold, negative_threshold=negative_threshold, cutoff_freq=cutoff_freq, full_scale_count=full_scale_count)
vid.event_simulator()
vid.render_video()