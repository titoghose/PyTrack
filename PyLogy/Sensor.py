class Sensor:

	sensor_names = ["EyeTracker"]
	
	meta_cols = {"EyeTracker": ["response_time", "pupil_size", "time_to_peak_pupil", "peak_pupil", "pupil_mean", "pupil_size_downsample", "pupil_slope", "pupil_area_curve", "blink_rate", "peak_blink_duration", "avg_blink_duration", "fixation_count", "max_fixation_duration", "avg_fixation_duration", "sacc_count", "sacc_duration", "sacc_vel", "sacc_amplitude", "ms_count", "ms_duration", "ms_vel", "ms_amplitude"]}

	def __init__(self, name, sampling_freq):
		self.name = name 
		self.metadata = {key:None for key in self.meta_cols[name]}
		self.sampling_freq = sampling_freq

