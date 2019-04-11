class Sensor:

	sensor_names = ["EyeTracker", "EEG"]
	
	meta_cols = {"EyeTracker": ["response_time", "pupil_size", "time_to_peak_pupil", "peak_pupil", "pupil_mean", "pupil_size_downsample", "pupil_slope", "pupil_area_curve", "blink_rate", "peak_blink_duration", "avg_blink_duration", "fixation_count", "max_fixation_duration", "avg_fixation_duration", "sacc_count", "sacc_duration", "sacc_vel", "sacc_amplitude", "ms_count", "ms_duration", "ms_vel", "ms_amplitude"], "EEG" : []}

	eeg_montage = {"standard_1020" : ['LPA', 'RPA', 'Nz', 'Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', 'O1', 'Oz', 'O2', 'O9', 'Iz', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2']}

	def __init__(self, name, sampling_freq):
		self.name = name 
		self.metadata = {key:None for key in self.meta_cols[name]}
		self.sampling_freq = sampling_freq

